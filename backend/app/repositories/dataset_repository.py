from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from app import config


class DatasetRepository:
    @staticmethod
    @lru_cache(maxsize=2)
    def _load_csv_cached(path: str) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_json_cached(path: str) -> dict:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_dataset_paths(self) -> dict:
        return {
            "match_dataset": config.MATCH_DATASET_PATH,
            "match_feature_sets": config.MATCH_FEATURE_SETS_PATH,
            "player_points_dataset": config.PLAYER_POINTS_DATASET_PATH,
            "player_points_feature_sets": config.PLAYER_POINTS_FEATURE_SETS_PATH,
        }

    def load_match_dataset(self) -> tuple[pd.DataFrame | None, str | None]:
        path = Path(config.MATCH_DATASET_PATH)
        if not path.exists():
            return None, f"Match dataset not found: {path}"
        try:
            return self._load_csv_cached(str(path)), None
        except Exception as exc:
            return None, f"Failed to load match dataset {path.name}: {exc}"

    def load_player_points_dataset(self) -> tuple[pd.DataFrame | None, str | None]:
        path = Path(config.PLAYER_POINTS_DATASET_PATH)
        if not path.exists():
            return None, f"Player points dataset not found: {path}"
        try:
            return self._load_csv_cached(str(path)), None
        except Exception as exc:
            return None, f"Failed to load player points dataset {path.name}: {exc}"

    def load_match_feature_sets(self) -> tuple[dict | None, str | None]:
        path = Path(config.MATCH_FEATURE_SETS_PATH)
        if not path.exists():
            return None, f"Match feature sets not found: {path}"
        try:
            return self._load_json_cached(str(path)), None
        except Exception as exc:
            return None, f"Failed to load match feature sets {path.name}: {exc}"

    def load_player_points_feature_sets(self) -> tuple[dict | None, str | None]:
        path = Path(config.PLAYER_POINTS_FEATURE_SETS_PATH)
        if not path.exists():
            return None, f"Player points feature sets not found: {path}"
        try:
            return self._load_json_cached(str(path)), None
        except Exception as exc:
            return None, f"Failed to load player points feature sets {path.name}: {exc}"
