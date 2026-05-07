from __future__ import annotations

import re
from typing import Any

import pandas as pd


RU_COLUMN_MAP = {
    "player": "Игрок",
    "player_name": "Игрок",
    "team": "Команда",
    "team_name": "Команда",
    "opponent": "Соперник",
    "opponent_team_name": "Соперник",
    "season": "Сезон",
    "game_date": "Дата матча",
    "date": "Дата матча",
    "points": "Очки",
    "target_points": "Фактические очки",
    "prediction": "Прогноз",
    "target_win": "Победа",
    "predicted_win": "Прогноз победы",
    "predicted_points": "Прогноз очков",
    "home_team": "Домашняя команда",
    "away_team": "Гостевая команда",
    "games_count": "Количество матчей",
    "wins": "Победы",
    "losses": "Поражения",
    "win_rate": "Доля побед",
    "minutes": "Минуты",
    "game_id": "Идентификатор матча",
    "player_id": "Идентификатор игрока",
    "team_id": "Идентификатор команды",
    "news_count": "Количество новостей",
    "news_sentiment_score": "Тональность новостей",
    "title": "Заголовок новости",
    "headline": "Заголовок новости",
    "published_at": "Дата публикации",
    "news_date": "Дата публикации",
    "source": "Источник",
}


def _humanize_feature_name(name: str) -> str:
    tokens = name.split("_")
    if len(tokens) == 1:
        return name
    return " ".join(tokens).strip()


def safe_user_error(payload: dict | None, default_text: str) -> str:
    if not payload:
        return default_text
    detail = str(payload.get("detail") or payload.get("error") or "").lower()
    if "model" in detail or "sklearn" in detail or "joblib" in detail:
        return "Модель временно недоступна. Проверьте наличие файла модели и зависимости проекта."
    if "not found" in detail:
        return "Запрошенные данные не найдены."
    return default_text


def expand_features_column(df: pd.DataFrame, features_col: str = "features") -> pd.DataFrame:
    if features_col not in df.columns:
        return df
    payload = df[features_col].apply(lambda x: x if isinstance(x, dict) else {})
    features_df = pd.json_normalize(payload)
    features_df.columns = [f"feature_{c}" for c in features_df.columns]
    out = pd.concat([df.drop(columns=[features_col]), features_df], axis=1)
    return out


def localize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in RU_COLUMN_MAP:
            rename_map[col] = RU_COLUMN_MAP[col]
            continue
        if col.startswith("feature_"):
            raw_name = col.replace("feature_", "", 1)
            rename_map[col] = f"Признак: {RU_COLUMN_MAP.get(raw_name, _humanize_feature_name(raw_name))}"
            continue
        rename_map[col] = RU_COLUMN_MAP.get(col, col)
    return df.rename(columns=rename_map)


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    out = expand_features_column(df)
    out = localize_dataframe_columns(out)
    return out


def normalize_team_code(team_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(team_name or "").upper())
    return cleaned[:3]
