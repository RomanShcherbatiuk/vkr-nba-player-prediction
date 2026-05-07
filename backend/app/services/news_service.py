from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from app import config


class NewsService:
    def __init__(self) -> None:
        self._data_root = Path(config.PROJECT_ROOT) / "data"
        self._daily_features_path = self._data_root / "nba_news_daily_features.csv"
        self._clean_news_path = self._data_root / "nba_news_clean.parquet"

    @staticmethod
    def _normalize_team(team_name: str) -> str:
        raw = "".join(ch for ch in str(team_name or "").upper() if ch.isalnum())
        return raw[:3]

    @lru_cache(maxsize=1)
    def load_daily_features(self) -> pd.DataFrame:
        if not self._daily_features_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self._daily_features_path, low_memory=False)
            if "news_date" in df.columns:
                df["news_date"] = pd.to_datetime(df["news_date"], errors="coerce")
            if "team" in df.columns:
                df["team_norm"] = df["team"].astype(str).map(self._normalize_team)
            return df
        except Exception:
            return pd.DataFrame()

    @lru_cache(maxsize=1)
    def load_clean_news(self) -> pd.DataFrame:
        if not self._clean_news_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(self._clean_news_path)
            if "published_at" in df.columns:
                df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            team_col = None
            for c in ["team", "team_code", "team_abbr", "abbr"]:
                if c in df.columns:
                    team_col = c
                    break
            if team_col:
                df["team_norm"] = df[team_col].astype(str).map(self._normalize_team)
            return df
        except Exception:
            return pd.DataFrame()

    def get_team_news_items(self, team_name: str, limit: int = 20) -> list[dict]:
        team_norm = self._normalize_team(team_name)
        out: list[dict] = []

        daily = self.load_daily_features()
        if not daily.empty and "team_norm" in daily.columns:
            subset = daily[daily["team_norm"] == team_norm].sort_values("news_date", ascending=False).head(limit)
            for _, row in subset.iterrows():
                out.append(
                    {
                        "date": None if pd.isna(row.get("news_date")) else str(pd.to_datetime(row["news_date"]).date()),
                        "team": row.get("team"),
                        "news_count": None if pd.isna(row.get("news_count")) else float(row.get("news_count")),
                        "news_sentiment_score": None
                        if pd.isna(row.get("avg_sentiment_score_kw"))
                        else float(row.get("avg_sentiment_score_kw")),
                        "tone_sentiment_features": {
                            "avg_gdelt_tone": row.get("avg_gdelt_tone"),
                            "positive_news_count": row.get("positive_news_count"),
                            "negative_news_count": row.get("negative_news_count"),
                            "injury_news_count": row.get("injury_news_count"),
                        },
                    }
                )

        clean = self.load_clean_news()
        if not clean.empty and "team_norm" in clean.columns:
            subset = clean[clean["team_norm"] == team_norm].copy()
            date_col = "published_at" if "published_at" in subset.columns else None
            title_col = "title" if "title" in subset.columns else ("headline" if "headline" in subset.columns else None)
            source_col = "source" if "source" in subset.columns else None
            if date_col:
                subset = subset.sort_values(date_col, ascending=False)
            for _, row in subset.head(limit).iterrows():
                out.append(
                    {
                        "date": None if not date_col or pd.isna(row.get(date_col)) else str(pd.to_datetime(row.get(date_col)).date()),
                        "team": team_name,
                        "title": row.get(title_col) if title_col else None,
                        "source": row.get(source_col) if source_col else None,
                    }
                )
        return out[:limit]
