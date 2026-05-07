from fastapi import APIRouter, HTTPException, Query

from app.services.match_service import MatchService, MatchServiceError

router = APIRouter()
match_service = MatchService()


@router.get("/")
def list_matches(
    team_id: str | None = Query(default=None),
    season: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    try:
        return match_service.list_matches(team_id=team_id, season=season, limit=limit)
    except MatchServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
