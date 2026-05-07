from app.api.artifacts import router as artifacts_router
from fastapi import FastAPI

from app.api.matches import router as matches_router
from app.api.methodology import router as methodology_router
from app.api.players import router as players_router
from app.api.predictions import router as predictions_router
from app.api.teams import router as teams_router


app = FastAPI(
    title="NBA Performance App API",
    description="Backend API for NBA thesis demo app.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(players_router, prefix="/api/v1/players", tags=["players"])
app.include_router(teams_router, prefix="/api/v1/teams", tags=["teams"])
app.include_router(matches_router, prefix="/api/v1/matches", tags=["matches"])
app.include_router(predictions_router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(methodology_router, prefix="/api/v1/methodology", tags=["methodology"])
app.include_router(artifacts_router, prefix="/api/v1/artifacts", tags=["artifacts"])
