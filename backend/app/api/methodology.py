from fastapi import APIRouter, HTTPException

from app.services.methodology_service import MethodologyService

router = APIRouter()
methodology_service = MethodologyService()


@router.get("/overview")
def methodology_overview() -> dict:
    try:
        return methodology_service.get_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build methodology overview: {exc}") from exc


@router.get("/model-cards")
def methodology_model_cards() -> dict:
    try:
        return methodology_service.get_model_cards()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build model cards: {exc}") from exc


@router.get("/metrics")
def methodology_metrics() -> dict:
    try:
        return methodology_service.get_metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {exc}") from exc


@router.get("/feature-groups")
def methodology_feature_groups() -> dict:
    try:
        return methodology_service.get_feature_groups()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load feature groups: {exc}") from exc
