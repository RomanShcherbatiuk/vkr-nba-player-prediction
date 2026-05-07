from pathlib import Path

from fastapi import APIRouter

from app import config
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository


router = APIRouter()


@router.get("/status")
def artifacts_status() -> dict:
    dataset_repo = DatasetRepository()
    artifact_repo = ArtifactRepository()
    model_repo = ModelRepository()

    dataset_paths = dataset_repo.get_dataset_paths()
    dataset_status = {}
    missing_files: list[str] = []
    for key, path in dataset_paths.items():
        exists = Path(path).exists()
        dataset_status[key] = {"path": path, "exists": exists}
        if not exists:
            missing_files.append(path)

    model_status = model_repo.list_available_models()
    missing_files.extend(model_status["missing"])

    reports = artifact_repo.load_available_reports()
    figures = artifact_repo.load_available_figures()
    metrics = artifact_repo.load_model_metrics()

    missing_files.extend(metrics["missing"].values())

    models = model_status["models"]
    models_ready = all(item["exists"] for item in models.values())
    model_metadata_ready = all(item["metadata_exists"] for item in models.values())

    app_ready = (
        dataset_status["match_dataset"]["exists"]
        and dataset_status["match_feature_sets"]["exists"]
        and dataset_status["player_points_dataset"]["exists"]
        and dataset_status["player_points_feature_sets"]["exists"]
        and models_ready
        and model_metadata_ready
    )

    return {
        "status": "ready" if app_ready else "partial",
        "project_root": str(config.PROJECT_ROOT),
        "datasets": dataset_status,
        "feature_sets": {
            "match": dataset_status["match_feature_sets"],
            "player_points": dataset_status["player_points_feature_sets"],
        },
        "models": model_status["models"],
        "reports": reports,
        "figures": figures,
        "metrics": metrics,
        "missing_files": sorted(set(missing_files)),
        "readiness": {
            "datasets_ready": all(item["exists"] for item in dataset_status.values()),
            "models_ready": models_ready,
            "model_metadata_ready": model_metadata_ready,
            "app_ready_for_inference": app_ready,
        },
    }
