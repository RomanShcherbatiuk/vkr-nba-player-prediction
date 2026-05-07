from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository
from app.services.news_service import NewsService
from app.utils.feature_utils import align_dataframe_to_model_features, build_missing_features_report


class TeamServiceError(Exception):
    pass


class TeamNotFoundError(TeamServiceError):
    pass


class TeamService:
    def __init__(self) -> None:
        self.dataset_repository = DatasetRepository()
        self.model_repository = ModelRepository()
        self.news_service = NewsService()

    @staticmethod
    def make_team_id(team_name: str) -> str:
        normalized = (team_name or "").strip().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        return f"tm_{digest}"

    @staticmethod
    def _resolve_column(columns: list[str], candidates: list[str]) -> str | None:
        by_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in by_lower:
                return by_lower[candidate.lower()]
        return None

    @staticmethod
    def _resolve_schema(df: pd.DataFrame) -> dict:
        cols = list(df.columns)
        schema = {
            "game_id": TeamService._resolve_column(cols, ["game_id"]),
            "date": TeamService._resolve_column(cols, ["game_date", "date"]),
            "season": TeamService._resolve_column(cols, ["season"]),
            "team": TeamService._resolve_column(cols, ["team"]),
            "opponent": TeamService._resolve_column(cols, ["opponent"]),
            "is_home": TeamService._resolve_column(cols, ["is_home", "home"]),
            "target_win": TeamService._resolve_column(cols, ["target_win", "win"]),
            "pts_for_proxy": TeamService._resolve_column(cols, ["team_pts_last5", "team_pts_last3"]),
            "pts_diff_proxy": TeamService._resolve_column(cols, ["team_pts_last5_diff", "team_pts_last3_diff"]),
        }
        if schema["team"] is None:
            raise TeamServiceError("Team column was not found in match dataset.")
        if schema["target_win"] is None:
            raise TeamServiceError("target_win column was not found in match dataset.")
        return schema

    @staticmethod
    def _serialize_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, pd.Timestamp):
            return str(value.date())
        return str(value)

    @staticmethod
    def _news_columns(columns: list[str]) -> list[str]:
        selected = []
        for col in columns:
            low = col.lower()
            if "news" in low or "gdelt" in low or "sentiment" in low or "tone" in low:
                if not col.startswith("opp_") and not col.endswith("_diff"):
                    selected.append(col)
        return selected

    @staticmethod
    def _main_feature_columns(columns: list[str]) -> list[str]:
        preferred = [
            "team_win_rate_last5",
            "team_pts_last5",
            "team_efficiency_proxy_last5",
            "roster_avg_age",
            "roster_avg_pts_per_game",
            "payroll",
            "news_count_lag1",
            "avg_gdelt_tone_lag1",
            "injury_news_count_lag1",
            "team_pts_last5_diff",
            "team_win_rate_last5_diff",
        ]
        return [c for c in preferred if c in columns]

    @staticmethod
    def _is_numeric_value(value) -> bool:
        return isinstance(value, (int, float)) and not pd.isna(value)

    @lru_cache(maxsize=1)
    def _dataset_schema(self) -> tuple[pd.DataFrame, dict]:
        df, error = self.dataset_repository.load_match_dataset()
        if error or df is None:
            raise TeamServiceError(error or "Match dataset is unavailable.")

        schema = self._resolve_schema(df)
        for col in [schema["target_win"], schema["is_home"], schema["pts_for_proxy"], schema["pts_diff_proxy"]]:
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if schema["date"] and schema["date"] in df.columns:
            df[schema["date"]] = pd.to_datetime(df[schema["date"]], errors="coerce")
        return df, schema

    @staticmethod
    def _latest_row_for_period(
        subset: pd.DataFrame,
        schema: dict,
        season: str | None = None,
        match_date: str | None = None,
    ) -> pd.Series | None:
        work = subset.copy()
        season_col = schema.get("season")
        date_col = schema.get("date")
        if season and season_col and season_col in work.columns:
            work = work[work[season_col].astype(str) == str(season)]
        if match_date and date_col and date_col in work.columns:
            dt = pd.to_datetime(match_date, errors="coerce")
            if not pd.isna(dt):
                work = work[work[date_col] <= dt]
        if work.empty:
            return None
        if date_col and date_col in work.columns:
            work = work.sort_values(date_col, ascending=False)
        return work.iloc[0]

    @staticmethod
    def _build_pair_feature_row(team_row: pd.Series, opponent_row: pd.Series, schema: dict) -> pd.Series:
        row = team_row.copy()
        team_col = schema.get("team")
        opponent_col = schema.get("opponent")
        if team_col:
            row[team_col] = team_row.get(team_col)
        if opponent_col:
            row[opponent_col] = opponent_row.get(team_col)

        for col in list(row.index):
            if col.startswith("opp_"):
                base_col = col[4:]
                if base_col in opponent_row.index:
                    row[col] = opponent_row.get(base_col)
            elif col.endswith("_diff"):
                base_col = col[:-5]
                opp_col = f"opp_{base_col}"
                team_val = row.get(base_col)
                opp_val = row.get(opp_col)
                if TeamService._is_numeric_value(team_val) and TeamService._is_numeric_value(opp_val):
                    row[col] = float(team_val) - float(opp_val)
        return row

    @lru_cache(maxsize=1)
    def _teams_index(self) -> pd.DataFrame:
        df, schema = self._dataset_schema()
        team_col = schema["team"]
        win_col = schema["target_win"]
        date_col = schema["date"]

        work = df[[c for c in [team_col, win_col, date_col] if c in df.columns]].copy()
        if date_col and date_col in work.columns:
            work = work.sort_values(date_col)

        grouped = work.groupby(team_col, dropna=False)
        out = grouped.agg(
            games_count=(team_col, "size"),
            wins=(win_col, lambda s: int((s == 1).sum())),
            losses=(win_col, lambda s: int((s == 0).sum())),
            win_rate=(win_col, "mean"),
            last_game_date=(date_col, "max") if date_col else (team_col, "first"),
        ).reset_index()
        out = out.rename(columns={team_col: "team_name"})
        out["team_name"] = out["team_name"].astype(str)
        out["team_id"] = out["team_name"].map(self.make_team_id)
        return out

    def _team_name_by_id(self, team_id: str) -> str:
        idx = self._teams_index()
        hit = idx[idx["team_id"] == team_id]
        if hit.empty:
            raise TeamNotFoundError(f"Team with id '{team_id}' was not found.")
        return str(hit.iloc[0]["team_name"])

    def list_teams(self, q: str | None = None, limit: int = 200) -> list[dict]:
        idx = self._teams_index().copy()
        if q:
            ql = q.strip().lower()
            idx = idx[idx["team_name"].str.lower().str.contains(ql, na=False)]
        idx = idx.sort_values("team_name").head(limit)
        items = []
        for _, row in idx.iterrows():
            items.append(
                {
                    "team_id": row["team_id"],
                    "team_name": row["team_name"],
                    "games_count": int(row["games_count"]),
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "win_rate": None if pd.isna(row["win_rate"]) else float(round(row["win_rate"], 4)),
                    "last_game_date": None
                    if pd.isna(row["last_game_date"])
                    else str(pd.to_datetime(row["last_game_date"]).date()),
                }
            )
        return items

    def get_team_card(self, team_id: str) -> dict:
        team_name = self._team_name_by_id(team_id)
        df, schema = self._dataset_schema()
        subset = df[df[schema["team"]].astype(str) == team_name].copy()
        if subset.empty:
            raise TeamNotFoundError(f"Team '{team_name}' has no rows in dataset.")
        if schema["date"]:
            subset = subset.sort_values(schema["date"], ascending=False)

        wins = int((subset[schema["target_win"]] == 1).sum())
        losses = int((subset[schema["target_win"]] == 0).sum())
        games_count = int(len(subset))
        win_rate = float(round(wins / games_count, 4)) if games_count else None

        avg_points_for = None
        avg_points_against = None
        if schema["pts_for_proxy"] and schema["pts_for_proxy"] in subset.columns:
            v = subset[schema["pts_for_proxy"]].mean()
            avg_points_for = None if pd.isna(v) else float(round(v, 3))
        if (
            schema["pts_for_proxy"]
            and schema["pts_diff_proxy"]
            and schema["pts_for_proxy"] in subset.columns
            and schema["pts_diff_proxy"] in subset.columns
        ):
            against_series = subset[schema["pts_for_proxy"]] - subset[schema["pts_diff_proxy"]]
            v = against_series.mean()
            avg_points_against = None if pd.isna(v) else float(round(v, 3))

        last_games = self.get_team_games(team_id=team_id, limit=5)["items"]
        return {
            "team_id": team_id,
            "team_name": team_name,
            "games_count": games_count,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_points_for": avg_points_for,
            "avg_points_against": avg_points_against,
            "last_games": last_games,
        }

    def get_team_games(self, team_id: str, season: str | None = None, limit: int = 50) -> dict:
        team_name = self._team_name_by_id(team_id)
        df, schema = self._dataset_schema()
        subset = df[df[schema["team"]].astype(str) == team_name].copy()
        if season and schema["season"] and schema["season"] in subset.columns:
            subset = subset[subset[schema["season"]].astype(str) == str(season)]
        if schema["date"]:
            subset = subset.sort_values(schema["date"], ascending=False)
        subset = subset.head(limit)

        main_features = self._main_feature_columns(list(subset.columns))
        items = []
        for _, row in subset.iterrows():
            feature_values = {}
            for feature in main_features:
                feature_values[feature] = self._serialize_value(row.get(feature))
            items.append(
                {
                    "date": self._serialize_value(row.get(schema["date"])) if schema["date"] else None,
                    "team": self._serialize_value(row.get(schema["team"])),
                    "opponent": self._serialize_value(row.get(schema["opponent"])) if schema["opponent"] else None,
                    "is_home": self._serialize_value(row.get(schema["is_home"])) if schema["is_home"] else None,
                    "target_win": self._serialize_value(row.get(schema["target_win"])),
                    "game_id": self._serialize_value(row.get(schema["game_id"])) if schema["game_id"] else None,
                    "features": feature_values,
                }
            )
        return {"team_id": team_id, "team_name": team_name, "items": items}

    def get_team_news(self, team_id: str, limit: int = 50) -> dict:
        team_name = self._team_name_by_id(team_id)
        df, schema = self._dataset_schema()
        subset = df[df[schema["team"]].astype(str) == team_name].copy()
        if schema["date"]:
            subset = subset.sort_values(schema["date"], ascending=False)
        subset = subset.head(limit)

        news_cols = self._news_columns(list(subset.columns))
        if not news_cols:
            return {"team_id": team_id, "team_name": team_name, "items": [], "message": "No news features available."}

        sentiment_col = None
        count_col = None
        for col in news_cols:
            if col == "avg_sentiment_score_kw_lag1":
                sentiment_col = col
            if col == "news_count_lag1":
                count_col = col

        items = []
        for _, row in subset.iterrows():
            tone_payload = {}
            for col in news_cols:
                tone_payload[col] = self._serialize_value(row.get(col))
            items.append(
                {
                    "date": self._serialize_value(row.get(schema["date"])) if schema["date"] else None,
                    "team": team_name,
                    "news_sentiment_score": self._serialize_value(row.get(sentiment_col)) if sentiment_col else None,
                    "news_count": self._serialize_value(row.get(count_col)) if count_col else None,
                    "tone_sentiment_features": tone_payload,
                }
            )
        external_news = self.news_service.get_team_news_items(team_name=team_name, limit=limit)
        if external_news:
            items = items + external_news
        return {"team_id": team_id, "team_name": team_name, "items": items}

    @staticmethod
    def _top_factors_from_model(model, aligned_df: pd.DataFrame, feature_list: list[str], top_n: int = 10) -> tuple[list[dict], str]:
        row = aligned_df.iloc[0]
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_")
            factors = []
            for idx, feature_name in enumerate(feature_list):
                if idx < len(importances):
                    importance = float(importances[idx])
                    value = float(row[feature_name])
                    factors.append(
                        {
                            "feature_name": feature_name,
                            "importance": importance,
                            "value": value,
                            "contribution_proxy": float(abs(importance) * abs(value)),
                        }
                    )
            factors.sort(key=lambda x: x["contribution_proxy"], reverse=True)
            return factors[:top_n], "model_feature_importances"

        if hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            if hasattr(coef, "ravel"):
                coef = coef.ravel()
            factors = []
            for idx, feature_name in enumerate(feature_list):
                if idx < len(coef):
                    c = float(coef[idx])
                    v = float(row[feature_name])
                    factors.append(
                        {
                            "feature_name": feature_name,
                            "coefficient": c,
                            "value": v,
                            "contribution_proxy": float(abs(c) * abs(v)),
                        }
                    )
            factors.sort(key=lambda x: x["contribution_proxy"], reverse=True)
            return factors[:top_n], "model_coefficients"

        return [], "not_available"

    def get_team_prediction(self, team_id: str) -> dict:
        team_name = self._team_name_by_id(team_id)
        df, schema = self._dataset_schema()
        subset = df[df[schema["team"]].astype(str) == team_name].copy()
        if subset.empty:
            raise TeamNotFoundError(f"Team '{team_name}' has no rows in dataset.")
        if schema["date"]:
            subset = subset.sort_values(schema["date"], ascending=False)
        source_row = subset.iloc[0]

        bundle = self.model_repository.load_match_winner_model_bundle()
        if bundle.get("status") != "ok":
            return {
                "status": "model_unavailable",
                "team_id": team_id,
                "team_name": team_name,
                "model_mode": "real",
                "error": "Модель временно недоступна. Проверьте наличие файла модели и зависимости проекта.",
                "technical_error": bundle.get("error"),
                "selection_source": bundle.get("selection_source"),
                "model_path": bundle.get("model_path"),
                "model_metadata_available": False,
            }

        model = bundle["model"]
        feature_list = bundle.get("feature_list") or []
        if not feature_list:
            feature_sets, _ = self.dataset_repository.load_match_feature_sets()
            if feature_sets:
                feature_list = feature_sets.get("all_model_features", [])

        aligned_df, feature_report = align_dataframe_to_model_features(
            pd.DataFrame([source_row]),
            feature_list=feature_list,
            fill_value=0.0,
        )
        missing_report = build_missing_features_report(feature_report)

        try:
            predicted_raw = model.predict(aligned_df)
            predicted_win = int(predicted_raw[0])
        except Exception as exc:
            return {
                "status": "model_unavailable",
                "team_id": team_id,
                "team_name": team_name,
                "model_mode": "real",
                "error": "Не удалось выполнить прогноз. Проверьте корректность модели и входных признаков.",
                "technical_error": str(exc),
                "missing_features": missing_report,
            }

        probability_available = hasattr(model, "predict_proba")
        win_probability = None
        if probability_available:
            try:
                proba = model.predict_proba(aligned_df)
                if len(proba.shape) == 2 and proba.shape[1] >= 2:
                    win_probability = float(proba[0][1])
                elif len(proba.shape) == 2 and proba.shape[1] == 1:
                    win_probability = float(proba[0][0])
                else:
                    probability_available = False
            except Exception:
                probability_available = False

        opponent_name = self._serialize_value(source_row.get(schema["opponent"])) if schema["opponent"] else None
        predicted_winner = team_name if predicted_win == 1 else opponent_name

        top_factors, explainability_method = self._top_factors_from_model(
            model=model,
            aligned_df=aligned_df,
            feature_list=list(aligned_df.columns),
            top_n=10,
        )

        metadata = bundle.get("metadata") or {}
        model_version = None
        if isinstance(metadata, dict):
            model_version = metadata.get("model_version") or metadata.get("version")

        source_date = source_row[schema["date"]] if schema["date"] else None
        return {
            "status": "ok",
            "team_id": team_id,
            "team_name": team_name,
            "source_row_date": None if pd.isna(source_date) else str(pd.to_datetime(source_date).date()),
            "opponent_team_name": opponent_name,
            "predicted_win": predicted_win,
            "predicted_winner": predicted_winner,
            "win_probability": win_probability,
            "probability_available": bool(probability_available and win_probability is not None),
            "model_name": bundle.get("model_name"),
            "model_type": bundle.get("model_type"),
            "model_version": model_version,
            "model_mode": "real",
            "model_path": bundle.get("model_path"),
            "model_metadata_available": bool(bundle.get("model_metadata_available")),
            "metrics": bundle.get("metrics", {}),
            "features_used_count": int(aligned_df.shape[1]),
            "features_used": list(aligned_df.columns),
            "missing_features": missing_report,
            "top_factors": top_factors,
            "explainability_method": explainability_method,
        }

    def get_matchup_prediction(
        self,
        team_1_id: str,
        team_2_id: str,
        season: str | None = None,
        match_date: str | None = None,
    ) -> dict:
        if team_1_id == team_2_id:
            raise TeamServiceError("Выберите две разные команды.")

        team_1_name = self._team_name_by_id(team_1_id)
        team_2_name = self._team_name_by_id(team_2_id)
        df, schema = self._dataset_schema()
        team_col = schema["team"]

        team_1_subset = df[df[team_col].astype(str) == team_1_name].copy()
        team_2_subset = df[df[team_col].astype(str) == team_2_name].copy()
        if team_1_subset.empty or team_2_subset.empty:
            raise TeamNotFoundError("Данные по выбранным командам отсутствуют.")

        team_1_row = self._latest_row_for_period(team_1_subset, schema=schema, season=season, match_date=match_date)
        team_2_row = self._latest_row_for_period(team_2_subset, schema=schema, season=season, match_date=match_date)
        if team_1_row is None or team_2_row is None:
            raise TeamServiceError("Недостаточно данных для выбранного периода.")

        pair_row = self._build_pair_feature_row(team_row=team_1_row, opponent_row=team_2_row, schema=schema)

        bundle = self.model_repository.load_match_winner_model_bundle()
        if bundle.get("status") != "ok":
            return {
                "status": "model_unavailable",
                "error": "Модель временно недоступна. Повторите попытку позже.",
            }

        model = bundle["model"]
        feature_list = bundle.get("feature_list") or []
        if not feature_list:
            feature_sets, _ = self.dataset_repository.load_match_feature_sets()
            if feature_sets:
                feature_list = feature_sets.get("all_model_features", [])

        aligned_df, feature_report = align_dataframe_to_model_features(
            pd.DataFrame([pair_row]),
            feature_list=feature_list,
            fill_value=0.0,
        )

        try:
            predicted_raw = model.predict(aligned_df)
            predicted_win_team_1 = int(predicted_raw[0])
        except Exception:
            return {
                "status": "model_unavailable",
                "error": "Не удалось выполнить прогноз по выбранным данным.",
            }

        prob_team_1 = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(aligned_df)
                if len(proba.shape) == 2 and proba.shape[1] >= 2:
                    prob_team_1 = float(proba[0][1])
                elif len(proba.shape) == 2 and proba.shape[1] == 1:
                    prob_team_1 = float(proba[0][0])
            except Exception:
                prob_team_1 = None

        if prob_team_1 is None:
            prob_team_1 = float(1.0 if predicted_win_team_1 == 1 else 0.0)
        prob_team_1 = max(0.0, min(1.0, prob_team_1))
        prob_team_2 = float(1.0 - prob_team_1)
        predicted_winner = team_1_name if prob_team_1 >= prob_team_2 else team_2_name

        top_factors, _ = self._top_factors_from_model(
            model=model,
            aligned_df=aligned_df,
            feature_list=list(aligned_df.columns),
            top_n=5,
        )
        key_feature_names = [x.get("feature_name") for x in top_factors if x.get("feature_name")]

        main_cols = self._main_feature_columns(list(df.columns))
        team_1_snapshot = {k: self._serialize_value(team_1_row.get(k)) for k in main_cols}
        team_2_snapshot = {k: self._serialize_value(team_2_row.get(k)) for k in main_cols}

        key_diffs = []
        for col in main_cols[:8]:
            v1 = team_1_row.get(col)
            v2 = team_2_row.get(col)
            if self._is_numeric_value(v1) and self._is_numeric_value(v2):
                key_diffs.append({"feature_name": col, "team_1_value": float(v1), "team_2_value": float(v2), "diff": float(v1) - float(v2)})

        return {
            "status": "ok",
            "team_1": {"team_id": team_1_id, "team_name": team_1_name, "latest_features": team_1_snapshot},
            "team_2": {"team_id": team_2_id, "team_name": team_2_name, "latest_features": team_2_snapshot},
            "season": season,
            "match_date": match_date,
            "predicted_winner": predicted_winner,
            "probability_team_1": float(round(prob_team_1, 4)),
            "probability_team_2": float(round(prob_team_2, 4)),
            "summary_ru": (
                f"По расчету модели преимущество у команды «{predicted_winner}». "
                f"Вероятность победы первой команды: {prob_team_1:.1%}, второй команды: {prob_team_2:.1%}."
            ),
            "key_model_features": key_feature_names,
            "key_feature_differences": key_diffs,
            "missing_features": build_missing_features_report(feature_report),
        }
