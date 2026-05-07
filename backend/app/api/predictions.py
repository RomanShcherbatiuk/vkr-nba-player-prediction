from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def prediction_info() -> dict:
    return {"message": "Predictions endpoint scaffold"}
