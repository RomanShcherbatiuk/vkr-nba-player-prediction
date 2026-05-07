from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.repositories.dataset_repository import DatasetRepository
from app.services.team_service import TeamService


class MatchServiceError(Exception):
    pass


class MatchService:
    def __init__(self) -> None:
        self.dataset_repository = DatasetRepository()

    @staticmethod
    def _resolve_column(columns: list[str], candidates: list[str]) -> str | None:
        by_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in by_lower:
                return by_lower[candidate.lower()]
        return None

    @staticmethod
    def _serialize(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, pd.Timestamp):
            return str(value.date())
        return str(value)

    @lru_cache(maxsize=1)
    def _dataset_schema(self) -> tuple[pd.DataFrame, dict]:
        df, error = self.dataset_repository.load_match_dataset()
        if error or df is None:
            raise MatchServiceError(error or "Match dataset is unavailable.")
        schema = {
            "game_id": self._resolve_column(list(df.columns), ["game_id"]),
            "date": self._resolve_column(list(df.columns), ["game_date", "date"]),
            "season": self._resolve_column(list(df.columns), ["season"]),
            "team": self._resolve_column(list(df.columns), ["team"]),
            "opponent": self._resolve_column(list(df.columns), ["opponent"]),
            "is_home": self._resolve_column(list(df.columns), ["is_home"]),
            "target_win": self._resolve_column(list(df.columns), ["target_win"]),
        }
        if schema["team"] is None:
            raise MatchServiceError("Team column was not found in match dataset.")
        if schema["date"] and schema["date"] in df.columns:
            df[schema["date"]] = pd.to_datetime(df[schema["date"]], errors="coerce")
        if schema["target_win"] and schema["target_win"] in df.columns:
            df[schema["target_win"]] = pd.to_numeric(df[schema["target_win"]], errors="coerce")
        return df, schema

    def list_matches(self, team_id: str | None = None, season: str | None = None, limit: int = 100) -> dict:
        df, schema = self._dataset_schema()
        subset = df.copy()
        if team_id:
            target_team = None
            team_col = schema["team"]
            if team_col:
                teams = subset[team_col].dropna().astype(str).unique().tolist()
                for team_name in teams:
                    if TeamService.make_team_id(team_name) == team_id:
                        target_team = team_name
                        break
            if target_team is None:
                return {"items": [], "count": 0, "message": f"Unknown team_id: {team_id}"}
            subset = subset[subset[schema["team"]].astype(str) == target_team]

        if season and schema["season"] and schema["season"] in subset.columns:
            subset = subset[subset[schema["season"]].astype(str) == str(season)]
        if schema["date"]:
            subset = subset.sort_values(schema["date"], ascending=False)
        subset = subset.head(limit)

        items = []
        for _, row in subset.iterrows():
            team_name = self._serialize(row.get(schema["team"])) if schema["team"] else None
            items.append(
                {
                    "game_id": self._serialize(row.get(schema["game_id"])) if schema["game_id"] else None,
                    "date": self._serialize(row.get(schema["date"])) if schema["date"] else None,
                    "season": self._serialize(row.get(schema["season"])) if schema["season"] else None,
                    "team_id": TeamService.make_team_id(team_name) if team_name else None,
                    "team": team_name,
                    "opponent": self._serialize(row.get(schema["opponent"])) if schema["opponent"] else None,
                    "is_home": self._serialize(row.get(schema["is_home"])) if schema["is_home"] else None,
                    "target_win": self._serialize(row.get(schema["target_win"])) if schema["target_win"] else None,
                }
            )
        return {"items": items, "count": len(items)}
