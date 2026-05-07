from fastapi import APIRouter, HTTPException, Query

from app.services.team_service import TeamNotFoundError, TeamService, TeamServiceError

router = APIRouter()
team_service = TeamService()


@router.get("/")
def list_teams(
    q: str | None = Query(default=None, description="Case-insensitive team search."),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    try:
        items = team_service.list_teams(q=q, limit=limit)
        return {"items": items, "count": len(items)}
    except TeamServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{team_id}")
def get_team(team_id: str) -> dict:
    try:
        return team_service.get_team_card(team_id=team_id)
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TeamServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{team_id}/games")
def get_team_games(
    team_id: str,
    season: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    try:
        return team_service.get_team_games(team_id=team_id, season=season, limit=limit)
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TeamServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{team_id}/news")
def get_team_news(
    team_id: str,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    try:
        return team_service.get_team_news(team_id=team_id, limit=limit)
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TeamServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{team_id}/prediction")
def get_team_prediction(team_id: str) -> dict:
    try:
        return team_service.get_team_prediction(team_id=team_id)
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TeamServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/prediction/matchup")
def get_matchup_prediction(
    team_1_id: str = Query(..., description="Идентификатор первой команды."),
    team_2_id: str = Query(..., description="Идентификатор второй команды."),
    season: str | None = Query(default=None, description="Сезон матча."),
    match_date: str | None = Query(default=None, description="Дата матча в формате YYYY-MM-DD."),
) -> dict:
    try:
        return team_service.get_matchup_prediction(
            team_1_id=team_1_id,
            team_2_id=team_2_id,
            season=season,
            match_date=match_date,
        )
    except TeamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TeamServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
