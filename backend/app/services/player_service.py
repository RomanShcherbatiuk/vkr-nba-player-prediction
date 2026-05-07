from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository
from app.utils.feature_utils import align_dataframe_to_model_features, build_missing_features_report


class PlayerServiceError(Exception):
    pass


class PlayerNotFoundError(PlayerServiceError):
    pass


class PlayerService:
    def __init__(self) -> None:
        self.dataset_repository = DatasetRepository()
        self.model_repository = ModelRepository()

    @staticmethod
    def _make_player_id(player_name: str) -> str:
        normalized = (player_name or "").strip().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"plr_{digest}"

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
            "player": PlayerService._resolve_column(cols, ["player", "player_name"]),
            "team": PlayerService._resolve_column(cols, ["team", "team_name"]),
            "season": PlayerService._resolve_column(cols, ["season"]),
            "game_date": PlayerService._resolve_column(cols, ["game_date", "date", "game_dt"]),
            "opponent": PlayerService._resolve_column(cols, ["opponent", "opponent_team"]),
            "target_points": PlayerService._resolve_column(cols, ["target_points", "points"]),
            "minutes": PlayerService._resolve_column(cols, ["minutes", "mp"]),
            "game_id": PlayerService._resolve_column(cols, ["game_id"]),
            "rebounds": PlayerService._resolve_column(cols, ["rebounds", "reb", "trb"]),
            "assists": PlayerService._resolve_column(cols, ["assists", "ast"]),
        }
        if schema["player"] is None:
            raise PlayerServiceError("Player column was not found in player points dataset.")
        return schema

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            if col and col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def _build_feature_group_lookup(feature_sets: dict) -> dict:
        group_lookup: dict = {}
        for group_name, features in feature_sets.items():
            if group_name == "all_model_features" or not isinstance(features, list):
                continue
            for feature in features:
                group_lookup[str(feature)] = group_name
        return group_lookup

    @staticmethod
    def _infer_feature_description(feature_name: str) -> str:
        name = feature_name.lower()
        if "news" in name or "gdelt" in name or "sentiment" in name:
            return "News and sentiment signal."
        if "salary" in name or "payroll" in name:
            return "Financial context signal."
        if name.startswith("player_"):
            return "Player form and recent performance."
        if name.startswith("team_"):
            return "Team context signal."
        if name.startswith("opponent_"):
            return "Opponent context signal."
        if name.startswith("previous_"):
            return "Previous season player signal."
        if name.endswith("_diff"):
            return "Relative difference feature."
        return "Model feature from prepared dataset."

    @staticmethod
    def _serialize_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return str(value)

    @lru_cache(maxsize=1)
    def _dataset_and_schema(self) -> tuple[pd.DataFrame, dict]:
        df, error = self.dataset_repository.load_player_points_dataset()
        if error or df is None:
            raise PlayerServiceError(error or "Player points dataset is unavailable.")
        schema = self._resolve_schema(df)
        numeric_cols = [schema["target_points"], schema["minutes"], schema["rebounds"], schema["assists"]]
        df = self._coerce_numeric(df, [x for x in numeric_cols if x])
        if schema["game_date"] and schema["game_date"] in df.columns:
            df[schema["game_date"]] = pd.to_datetime(df[schema["game_date"]], errors="coerce")
        return df, schema

    @lru_cache(maxsize=1)
    def _feature_sets(self) -> dict:
        feature_sets, _ = self.dataset_repository.load_player_points_feature_sets()
        return feature_sets or {}

    @lru_cache(maxsize=1)
    def _player_index(self) -> pd.DataFrame:
        df, schema = self._dataset_and_schema()
        player_col = schema["player"]
        team_col = schema["team"]
        points_col = schema["target_points"]
        date_col = schema["game_date"]

        work = df[[c for c in [player_col, team_col, points_col, date_col] if c in df.columns]].copy()
        if date_col and date_col in work.columns:
            work = work.sort_values(date_col)
        grouped = work.groupby(player_col, dropna=False)
        summary = grouped.agg(
            games_count=(player_col, "size"),
            avg_points=(points_col, "mean") if points_col else (player_col, "size"),
            team_name=(team_col, "last") if team_col else (player_col, "first"),
            last_game_date=(date_col, "max") if date_col else (player_col, "first"),
        ).reset_index()
        summary = summary.rename(columns={player_col: "player_name"})
        summary["player_name"] = summary["player_name"].astype(str)
        summary["player_id"] = summary["player_name"].map(self._make_player_id)
        return summary

    def _get_player_name_by_id(self, player_id: str) -> str:
        index = self._player_index()
        matched = index[index["player_id"] == player_id]
        if matched.empty:
            raise PlayerNotFoundError(f"Player with id '{player_id}' was not found.")
        return str(matched.iloc[0]["player_name"])

    def list_players(self, query: str | None = None, limit: int = 200) -> list[dict]:
        index = self._player_index().copy()
        if query:
            q = query.strip().lower()
            index = index[index["player_name"].str.lower().str.contains(q, na=False)]
        index = index.sort_values(["player_name"]).head(limit)
        rows: list[dict] = []
        for _, row in index.iterrows():
            last_game = row["last_game_date"]
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "team_name": None if pd.isna(row["team_name"]) else str(row["team_name"]),
                    "games_count": int(row["games_count"]),
                    "avg_points": None if pd.isna(row["avg_points"]) else float(round(row["avg_points"], 3)),
                    "last_game_date": None if pd.isna(last_game) else str(pd.to_datetime(last_game).date()),
                }
            )
        return rows

    def get_player_card(self, player_id: str) -> dict:
        player_name = self._get_player_name_by_id(player_id)
        df, schema = self._dataset_and_schema()
        subset = df[df[schema["player"]].astype(str) == player_name].copy()
        if subset.empty:
            raise PlayerNotFoundError(f"Player '{player_name}' has no rows in dataset.")

        subset = subset.sort_values(schema["game_date"], ascending=False) if schema["game_date"] else subset
        seasons = []
        if schema["season"] and schema["season"] in subset.columns:
            seasons = sorted([str(x) for x in subset[schema["season"]].dropna().unique().tolist()])

        team_name = None
        if schema["team"] and schema["team"] in subset.columns:
            team_value = subset[schema["team"]].dropna().head(1)
            team_name = None if team_value.empty else str(team_value.iloc[0])

        last_games = self.get_player_games(player_id=player_id, limit=5)["items"]

        def _mean_or_none(column_name: str | None):
            if not column_name or column_name not in subset.columns:
                return None
            value = subset[column_name].mean()
            return None if pd.isna(value) else float(round(value, 3))

        return {
            "player_id": player_id,
            "player_name": player_name,
            "team_name": team_name,
            "seasons": seasons,
            "games_count": int(len(subset)),
            "avg_points": _mean_or_none(schema["target_points"]),
            "avg_minutes": _mean_or_none(schema["minutes"]),
            "avg_rebounds": _mean_or_none(schema["rebounds"]),
            "avg_assists": _mean_or_none(schema["assists"]),
            "last_games_summary": last_games,
        }

    def get_player_games(self, player_id: str, season: str | None = None, limit: int = 50) -> dict:
        player_name = self._get_player_name_by_id(player_id)
        df, schema = self._dataset_and_schema()
        subset = df[df[schema["player"]].astype(str) == player_name].copy()
        if season and schema["season"] and schema["season"] in subset.columns:
            subset = subset[subset[schema["season"]].astype(str) == str(season)]
        if schema["game_date"]:
            subset = subset.sort_values(schema["game_date"], ascending=False)
        subset = subset.head(limit)

        key_feature_candidates = [
            "player_pts_last5",
            "player_reb_last5",
            "player_ast_last5",
            "team_win_rate_last5",
            "news_count_lag1",
        ]
        key_features = [c for c in key_feature_candidates if c in subset.columns]

        items = []
        for _, row in subset.iterrows():
            date_value = row[schema["game_date"]] if schema["game_date"] else None
            item = {
                "date": None if pd.isna(date_value) else str(pd.to_datetime(date_value).date()),
                "team": None if not schema["team"] or pd.isna(row.get(schema["team"])) else str(row[schema["team"]]),
                "opponent": None
                if not schema["opponent"] or pd.isna(row.get(schema["opponent"]))
                else str(row[schema["opponent"]]),
                "points": None
                if not schema["target_points"] or pd.isna(row.get(schema["target_points"]))
                else float(row[schema["target_points"]]),
                "minutes": None
                if not schema["minutes"] or pd.isna(row.get(schema["minutes"]))
                else float(row[schema["minutes"]]),
                "game_id": None if not schema["game_id"] or pd.isna(row.get(schema["game_id"])) else str(row[schema["game_id"]]),
                "features": {},
            }
            for feature in key_features:
                value = row.get(feature)
                item["features"][feature] = None if pd.isna(value) else float(value)
            items.append(item)

        return {"player_id": player_id, "player_name": player_name, "items": items}

    def get_player_features(
        self, player_id: str, game_id: str | None = None, season: str | None = None
    ) -> dict:
        player_name = self._get_player_name_by_id(player_id)
        df, schema = self._dataset_and_schema()
        subset = df[df[schema["player"]].astype(str) == player_name].copy()
        if season and schema["season"] and schema["season"] in subset.columns:
            subset = subset[subset[schema["season"]].astype(str) == str(season)]
        if subset.empty:
            raise PlayerNotFoundError(f"No rows found for player '{player_name}' with selected filters.")

        if game_id and schema["game_id"] and schema["game_id"] in subset.columns:
            matched = subset[subset[schema["game_id"]].astype(str) == str(game_id)]
            if not matched.empty:
                subset = matched

        if schema["game_date"]:
            subset = subset.sort_values(schema["game_date"], ascending=False)
        current_row = subset.iloc[0]

        feature_sets = self._feature_sets()
        all_features = feature_sets.get("all_model_features", [])
        group_lookup = self._build_feature_group_lookup(feature_sets)

        features: list[dict] = []
        for feature_name in all_features:
            if feature_name not in df.columns:
                continue
            value = current_row.get(feature_name)
            features.append(
                {
                    "feature_name": feature_name,
                    "current_value": self._serialize_value(value),
                    "feature_group": group_lookup.get(feature_name, "all_model_features"),
                    "description": self._infer_feature_description(feature_name),
                }
            )

        return {
            "player_id": player_id,
            "player_name": player_name,
            "game_id": game_id,
            "rows_considered": int(len(subset)),
            "features_count": len(features),
            "features": features,
        }

    @staticmethod
    def _explain_top_factors(model, aligned_df: pd.DataFrame, feature_list: list[str], top_n: int = 10) -> tuple[list[dict], str]:
        row = aligned_df.iloc[0]
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_")
            pairs = []
            for idx, feature_name in enumerate(feature_list):
                if idx < len(importances):
                    value = float(importances[idx])
                    pairs.append(
                        {
                            "feature_name": feature_name,
                            "importance": value,
                            "value": float(row[feature_name]),
                            "contribution_proxy": float(abs(value) * abs(float(row[feature_name]))),
                        }
                    )
            pairs.sort(key=lambda x: x["contribution_proxy"], reverse=True)
            return pairs[:top_n], "model_feature_importances"

        if hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            if hasattr(coef, "ravel"):
                coef = coef.ravel()
            pairs = []
            for idx, feature_name in enumerate(feature_list):
                if idx < len(coef):
                    c = float(coef[idx])
                    v = float(row[feature_name])
                    pairs.append(
                        {
                            "feature_name": feature_name,
                            "coefficient": c,
                            "value": v,
                            "contribution_proxy": float(abs(c) * abs(v)),
                        }
                    )
            pairs.sort(key=lambda x: x["contribution_proxy"], reverse=True)
            return pairs[:top_n], "model_coefficients"

        pairs = []
        for feature_name in feature_list:
            value = float(row[feature_name])
            pairs.append(
                {
                    "feature_name": feature_name,
                    "value": value,
                    "absolute_value": float(abs(value)),
                }
            )
        pairs.sort(key=lambda x: x["absolute_value"], reverse=True)
        return pairs[:top_n], "value_magnitude_proxy"

    def get_player_prediction(self, player_id: str) -> dict:
        player_name = self._get_player_name_by_id(player_id)
        df, schema = self._dataset_and_schema()
        subset = df[df[schema["player"]].astype(str) == player_name].copy()
        if subset.empty:
            raise PlayerNotFoundError(f"No rows found for player '{player_name}'.")
        if schema["game_date"]:
            subset = subset.sort_values(schema["game_date"], ascending=False)
        source_row = subset.iloc[0]

        bundle = self.model_repository.load_player_points_model_bundle()
        if bundle.get("status") != "ok":
            return {
                "status": "model_unavailable",
                "player_id": player_id,
                "player_name": player_name,
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
            feature_sets = self._feature_sets()
            feature_list = feature_sets.get("all_model_features", [])

        inference_df = pd.DataFrame([source_row])
        aligned_df, feature_report = align_dataframe_to_model_features(
            inference_df,
            feature_list=feature_list,
            fill_value=0.0,
        )
        missing_report = build_missing_features_report(feature_report)

        try:
            pred = model.predict(aligned_df)
            predicted_points = float(pred[0])
        except Exception as exc:
            return {
                "status": "model_unavailable",
                "player_id": player_id,
                "player_name": player_name,
                "model_mode": "real",
                "error": "Не удалось выполнить прогноз. Проверьте корректность модели и входных признаков.",
                "technical_error": str(exc),
                "missing_features": missing_report,
            }

        top_factors, explainability_method = self._explain_top_factors(
            model=model,
            aligned_df=aligned_df,
            feature_list=list(aligned_df.columns),
            top_n=10,
        )

        model_version = None
        metadata = bundle.get("metadata") or {}
        if isinstance(metadata, dict):
            model_version = metadata.get("model_version") or metadata.get("version")

        source_date = source_row[schema["game_date"]] if schema["game_date"] else None
        return {
            "status": "ok",
            "player_id": player_id,
            "player_name": player_name,
            "source_row_date": None if pd.isna(source_date) else str(pd.to_datetime(source_date).date()),
            "team_name": None if not schema["team"] else self._serialize_value(source_row.get(schema["team"])),
            "opponent": None if not schema["opponent"] else self._serialize_value(source_row.get(schema["opponent"])),
            "predicted_points": float(round(predicted_points, 3)),
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
            "explainability_method": explainability_method if top_factors else "not_available",
        }
