from pydantic import BaseModel


class PredictionRequest(BaseModel):
    game_id: str | None = None
    team: str | None = None
    opponent: str | None = None
    player: str | None = None


class PredictionResponse(BaseModel):
    status: str
    message: str
