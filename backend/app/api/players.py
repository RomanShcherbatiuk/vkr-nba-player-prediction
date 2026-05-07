from fastapi import APIRouter, HTTPException, Query

from app.services.player_service import PlayerNotFoundError, PlayerService, PlayerServiceError


router = APIRouter()
player_service = PlayerService()


@router.get("/")
def list_players(
    q: str | None = Query(default=None, description="Case-insensitive player name search."),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    try:
        items = player_service.list_players(query=q, limit=limit)
        return {"items": items, "count": len(items)}
    except PlayerServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{player_id}")
def get_player(player_id: str) -> dict:
    try:
        return player_service.get_player_card(player_id=player_id)
    except PlayerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlayerServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{player_id}/games")
def get_player_games(
    player_id: str,
    season: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    try:
        return player_service.get_player_games(player_id=player_id, season=season, limit=limit)
    except PlayerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlayerServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{player_id}/features")
def get_player_features(
    player_id: str,
    game_id: str | None = Query(default=None),
    season: str | None = Query(default=None),
) -> dict:
    try:
        return player_service.get_player_features(player_id=player_id, game_id=game_id, season=season)
    except PlayerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlayerServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{player_id}/prediction")
def get_player_prediction(player_id: str) -> dict:
    try:
        return player_service.get_player_prediction(player_id=player_id)
    except PlayerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlayerServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
