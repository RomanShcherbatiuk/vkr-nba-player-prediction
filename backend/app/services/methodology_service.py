from __future__ import annotations

from pathlib import Path

from app import config
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository


class MethodologyService:
    def __init__(self) -> None:
        self.dataset_repository = DatasetRepository()
        self.model_repository = ModelRepository()
        self.artifact_repository = ArtifactRepository()

    @staticmethod
    def _exists(path: str) -> bool:
        return Path(path).exists()

    def _load_baseline_doc(self) -> dict:
        return {
            "exists": False,
            "content_preview": "not_available",
        }

    def get_feature_groups(self) -> dict:
        match_feature_sets, match_error = self.dataset_repository.load_match_feature_sets()
        player_feature_sets, player_error = self.dataset_repository.load_player_points_feature_sets()

        return {
            "status": "ok",
            "match": match_feature_sets if match_feature_sets else "not_available",
            "player_points": player_feature_sets if player_feature_sets else "not_available",
            "errors": {
                "match_feature_sets": match_error or None,
                "player_points_feature_sets": player_error or None,
            },
        }

    def get_metrics(self) -> dict:
        metrics = self.artifact_repository.load_model_metrics()
        loaded = metrics.get("loaded", {})
        if not loaded:
            return {
                "status": "ok",
                "metrics": "not_available",
                "details": metrics,
            }
        return {
            "status": "ok",
            "metrics": loaded,
            "details": metrics,
        }

    def get_model_cards(self) -> dict:
        feature_groups = self.get_feature_groups()
        player_bundle = self.model_repository.load_player_points_model_bundle()
        match_bundle = self.model_repository.load_match_winner_model_bundle()

        player_metrics = player_bundle.get("metrics") if player_bundle.get("status") == "ok" else None
        match_metrics = match_bundle.get("metrics") if match_bundle.get("status") == "ok" else None

        player_card = {
            "task": "player_points_prediction",
            "target_variable": "target_points",
            "dataset": config.PLAYER_POINTS_DATASET_PATH,
            "feature_groups": (
                list(feature_groups["player_points"].keys())
                if isinstance(feature_groups.get("player_points"), dict)
                else "not_available"
            ),
            "model": player_bundle.get("model_name", "not_available"),
            "model_type": player_bundle.get("model_type", "not_available"),
            "metrics": player_metrics if player_metrics else "not_available",
            "limitations": [
                "Прогноз выполняется по последней доступной строке игрока.",
                "При отсутствии части признаков применяется безопасное заполнение.",
                "При недоступной модели возвращается model_unavailable.",
            ],
            "application_usage": "GET /api/v1/players/{player_id}/prediction",
        }

        match_card = {
            "task": "match_winner_prediction",
            "target_variable": "target_win",
            "dataset": config.MATCH_DATASET_PATH,
            "feature_groups": (
                list(feature_groups["match"].keys())
                if isinstance(feature_groups.get("match"), dict)
                else "not_available"
            ),
            "model": match_bundle.get("model_name", "not_available"),
            "model_type": match_bundle.get("model_type", "not_available"),
            "metrics": match_metrics if match_metrics else "not_available",
            "limitations": [
                "Прогноз выполняется по последней доступной строке команды.",
                "Вероятность доступна только при наличии predict_proba.",
                "При недоступной модели возвращается model_unavailable.",
            ],
            "application_usage": "GET /api/v1/teams/{team_id}/prediction",
        }

        return {
            "status": "ok",
            "player_points_model_card": player_card,
            "match_winner_model_card": match_card,
        }

    def get_overview(self) -> dict:
        dataset_paths = self.dataset_repository.get_dataset_paths()
        models_info = self.model_repository.list_available_models()
        metrics = self.get_metrics()
        baseline_doc = self._load_baseline_doc()

        return {
            "status": "ok",
            "thesis_title": "Предсказательные модели для анализа эффективности игроков в профессиональном баскетболе",
            "ml_tasks": [
                "player_points_regression",
                "match_winner_classification",
            ],
            "datasets": {
                key: {"path": path, "exists": self._exists(path)}
                for key, path in dataset_paths.items()
            },
            "models": models_info.get("models", {}),
            "metrics_availability": "not_available" if metrics.get("metrics") == "not_available" else "available",
            "baseline_models_doc": baseline_doc,
            "limitations": [
                "Frontend использует только backend API.",
                "Модели не переобучаются в приложении.",
                "При отсутствии артефактов возвращается статус model_unavailable.",
            ],
        }
