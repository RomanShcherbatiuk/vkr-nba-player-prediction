import os
from typing import Any

import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SEC = float(os.getenv("BACKEND_TIMEOUT_SEC", "10"))


def _safe_request(method: str, path: str, params: dict | None = None) -> tuple[bool, dict]:
    try:
        response = requests.request(
            method=method,
            url=f"{BACKEND_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT_SEC,
        )
        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_response": response.text}
        if not response.ok:
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            return False, {"error": f"HTTP {response.status_code}", "detail": detail}
        if isinstance(payload, dict):
            return True, payload
        return True, {"data": payload}
    except Exception as exc:
        return False, {"error": str(exc)}


def health_check() -> tuple[bool, dict]:
    return _safe_request("GET", "/health")


def get_artifacts_status() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/artifacts/status")


def get_players(query: str | None = None, limit: int = 500) -> tuple[bool, dict]:
    params = {"limit": limit}
    if query:
        params["q"] = query
    return _safe_request("GET", "/api/v1/players", params=params)


def get_player_card(player_id: str) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/players/{player_id}")


def get_player_games(player_id: str, season: str | None = None, limit: int = 100) -> tuple[bool, dict]:
    params = {"limit": limit}
    if season:
        params["season"] = season
    return _safe_request("GET", f"/api/v1/players/{player_id}/games", params=params)


def get_player_prediction(player_id: str) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/players/{player_id}/prediction")


def get_player_features(player_id: str) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/players/{player_id}/features")


def get_teams(query: str | None = None, limit: int = 500) -> tuple[bool, dict]:
    params = {"limit": limit}
    if query:
        params["q"] = query
    return _safe_request("GET", "/api/v1/teams", params=params)


def get_team_card(team_id: str) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/teams/{team_id}")


def get_team_games(team_id: str, season: str | None = None, limit: int = 100) -> tuple[bool, dict]:
    params = {"limit": limit}
    if season:
        params["season"] = season
    return _safe_request("GET", f"/api/v1/teams/{team_id}/games", params=params)


def get_team_news(team_id: str, limit: int = 100) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/teams/{team_id}/news", params={"limit": limit})


def get_team_prediction(team_id: str) -> tuple[bool, dict]:
    return _safe_request("GET", f"/api/v1/teams/{team_id}/prediction")


def get_matchup_prediction(
    team_1_id: str,
    team_2_id: str,
    season: str | None = None,
    match_date: str | None = None,
) -> tuple[bool, dict]:
    params: dict[str, Any] = {
        "team_1_id": team_1_id,
        "team_2_id": team_2_id,
    }
    if season:
        params["season"] = season
    if match_date:
        params["match_date"] = match_date
    return _safe_request("GET", "/api/v1/teams/prediction/matchup", params=params)


def get_matches(team_id: str | None = None, season: str | None = None, limit: int = 100) -> tuple[bool, dict]:
    params: dict[str, Any] = {"limit": limit}
    if team_id:
        params["team_id"] = team_id
    if season:
        params["season"] = season
    return _safe_request("GET", "/api/v1/matches", params=params)


def get_methodology_info() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/methodology/overview")


def get_methodology_overview() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/methodology/overview")


def get_methodology_model_cards() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/methodology/model-cards")


def get_methodology_metrics() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/methodology/metrics")


def get_methodology_feature_groups() -> tuple[bool, dict]:
    return _safe_request("GET", "/api/v1/methodology/feature-groups")
