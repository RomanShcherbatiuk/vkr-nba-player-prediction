import json
from pathlib import Path

import pandas as pd

from app import config
from app.utils.model_loader import ModelLoadError, load_model_safely


class ModelRepository:
    def expected_model_paths(self) -> dict:
        return {
            "match_winner_model": str(Path(config.MATCH_WINNER_MODEL_PATH)),
            "player_points_model": str(Path(config.PLAYER_POINTS_MODEL_PATH)),
        }

    def _expected_metadata_paths(self) -> dict:
        return {
            "match_winner_model": str(Path(config.MATCH_WINNER_MODEL_METADATA_PATH)),
            "player_points_model": str(Path(config.PLAYER_POINTS_MODEL_METADATA_PATH)),
        }

    @staticmethod
    def _sanitize_model_name_for_file_lookup(model_name: str) -> str:
        return "".join(ch.lower() for ch in model_name if ch.isalnum())

    def _load_player_results_tables(self) -> list[pd.DataFrame]:
        if not config.REPORTS_DIR:
            return []
        reports_dir = Path(config.REPORTS_DIR)
        files = [
            reports_dir / "player_points_regression_results.csv",
            reports_dir / "player_points_tuning_results.csv",
        ]
        tables = []
        for path in files:
            if path.exists():
                try:
                    tables.append(pd.read_csv(path))
                except Exception:
                    continue
        return tables

    def _load_match_results_tables(self) -> list[pd.DataFrame]:
        if not config.REPORTS_DIR:
            return []
        reports_dir = Path(config.REPORTS_DIR)
        files = [
            reports_dir / "match_classification_results.csv",
            reports_dir / "match_tuning_results.csv",
        ]
        tables = []
        for path in files:
            if path.exists():
                try:
                    tables.append(pd.read_csv(path))
                except Exception:
                    continue
        return tables

    def _select_best_player_model_name_from_tables(self) -> str | None:
        tables = self._load_player_results_tables()
        candidates: list[pd.DataFrame] = []
        for table in tables:
            work = table.copy()
            if "task" in work.columns:
                work = work[work["task"].astype(str).str.contains("player_points", case=False, na=False)]
            if "status" in work.columns:
                work = work[work["status"].astype(str).str.lower() == "trained"]
            if work.empty:
                continue
            candidates.append(work)
        if not candidates:
            return None

        merged = pd.concat(candidates, ignore_index=True)
        if "model" not in merged.columns:
            return None
        if "mae_test" in merged.columns:
            merged["__score"] = pd.to_numeric(merged["mae_test"], errors="coerce")
            merged = merged[merged["__score"].notna()].sort_values("__score", ascending=True)
            if not merged.empty:
                return str(merged.iloc[0]["model"])
        if "main_test_metric" in merged.columns:
            merged["__score"] = pd.to_numeric(merged["main_test_metric"], errors="coerce")
            merged = merged[merged["__score"].notna()].sort_values("__score", ascending=True)
            if not merged.empty:
                return str(merged.iloc[0]["model"])
        if "r2_test" in merged.columns:
            merged["__score"] = pd.to_numeric(merged["r2_test"], errors="coerce")
            merged = merged[merged["__score"].notna()].sort_values("__score", ascending=False)
            if not merged.empty:
                return str(merged.iloc[0]["model"])
        return None

    def _select_best_match_model_name_from_tables(self) -> str | None:
        tables = self._load_match_results_tables()
        candidates: list[pd.DataFrame] = []
        for table in tables:
            work = table.copy()
            if "task" in work.columns:
                work = work[
                    work["task"].astype(str).str.contains("match", case=False, na=False)
                    | work["task"].astype(str).str.contains("winner", case=False, na=False)
                ]
            if "status" in work.columns:
                work = work[work["status"].astype(str).str.lower() == "trained"]
            if work.empty:
                continue
            candidates.append(work)
        if not candidates:
            return None
        merged = pd.concat(candidates, ignore_index=True)
        if "model" not in merged.columns:
            return None
        if "roc_auc_test" in merged.columns:
            merged["__score"] = pd.to_numeric(merged["roc_auc_test"], errors="coerce")
            merged = merged[merged["__score"].notna()].sort_values("__score", ascending=False)
            if not merged.empty:
                return str(merged.iloc[0]["model"])
        if "main_test_metric" in merged.columns:
            merged["__score"] = pd.to_numeric(merged["main_test_metric"], errors="coerce")
            merged = merged[merged["__score"].notna()].sort_values("__score", ascending=False)
            if not merged.empty:
                return str(merged.iloc[0]["model"])
        return None

    def _discover_player_points_model_candidates(self) -> list[Path]:
        model_dir = Path(config.MODEL_ARTIFACTS_DIR)
        if not model_dir.exists():
            return []
        paths = []
        for pattern in ["*.joblib", "*.pkl"]:
            paths.extend(model_dir.glob(pattern))
        return sorted([p for p in paths if p.is_file()])

    def _discover_match_winner_model_candidates(self) -> list[Path]:
        model_dir = Path(config.MODEL_ARTIFACTS_DIR)
        if not model_dir.exists():
            return []
        paths = []
        for pattern in ["*.joblib", "*.pkl"]:
            paths.extend(model_dir.glob(pattern))
        return sorted([p for p in paths if p.is_file()])

    def _metadata_path_for_model_path(self, model_path: Path) -> Path:
        if model_path == Path(config.MATCH_WINNER_MODEL_PATH):
            return Path(config.MATCH_WINNER_MODEL_METADATA_PATH)
        if model_path == Path(config.PLAYER_POINTS_MODEL_PATH):
            return Path(config.PLAYER_POINTS_MODEL_METADATA_PATH)
        stem = model_path.stem
        return model_path.parent / f"{stem}_metadata.json"

    def _select_player_points_model_path(self) -> tuple[Path | None, str]:
        explicit = Path(config.PLAYER_POINTS_MODEL_PATH)
        if explicit.exists():
            return explicit, "env:PLAYER_POINTS_MODEL_PATH"

        best_model_name = self._select_best_player_model_name_from_tables()
        candidates = self._discover_player_points_model_candidates()
        if best_model_name and candidates:
            target = self._sanitize_model_name_for_file_lookup(best_model_name)
            for candidate in candidates:
                name_key = self._sanitize_model_name_for_file_lookup(candidate.name)
                if target in name_key:
                    return candidate, "reports:auto_best_by_metrics"

        for candidate in candidates:
            lowered = candidate.name.lower()
            if "player" in lowered and ("points" in lowered or "regression" in lowered):
                return candidate, "scan:player_points_pattern"

        if candidates:
            return candidates[0], "scan:first_available"
        return None, "not_found"

    def _select_match_winner_model_path(self) -> tuple[Path | None, str]:
        explicit = Path(config.MATCH_WINNER_MODEL_PATH)
        if explicit.exists():
            return explicit, "env:MATCH_WINNER_MODEL_PATH"

        best_model_name = self._select_best_match_model_name_from_tables()
        candidates = self._discover_match_winner_model_candidates()
        if best_model_name and candidates:
            target = self._sanitize_model_name_for_file_lookup(best_model_name)
            for candidate in candidates:
                name_key = self._sanitize_model_name_for_file_lookup(candidate.name)
                if target in name_key:
                    return candidate, "reports:auto_best_by_roc_auc"

        for candidate in candidates:
            lowered = candidate.name.lower()
            if "match" in lowered and ("winner" in lowered or "classification" in lowered):
                return candidate, "scan:match_winner_pattern"

        if candidates:
            return candidates[0], "scan:first_available"
        return None, "not_found"

    def _metrics_for_player_model_name(self, model_name: str | None) -> dict:
        tables = self._load_player_results_tables()
        if not tables:
            return {}
        merged = pd.concat(tables, ignore_index=True)
        if "model" in merged.columns and model_name:
            merged = merged[merged["model"].astype(str) == str(model_name)]
        if merged.empty:
            return {}
        row = merged.iloc[0].to_dict()
        metrics = {}
        for key in [
            "mae_val",
            "rmse_val",
            "r2_val",
            "mae_test",
            "rmse_test",
            "r2_test",
            "main_validation_metric",
            "main_test_metric",
        ]:
            if key in row and pd.notna(row[key]):
                try:
                    metrics[key] = float(row[key])
                except Exception:
                    metrics[key] = row[key]
        return metrics

    def _metrics_for_match_model_name(self, model_name: str | None) -> dict:
        tables = self._load_match_results_tables()
        if not tables:
            return {}
        merged = pd.concat(tables, ignore_index=True)
        if "model" in merged.columns and model_name:
            merged = merged[merged["model"].astype(str) == str(model_name)]
        if merged.empty:
            return {}
        row = merged.iloc[0].to_dict()
        metrics = {}
        for key in [
            "roc_auc_val",
            "f1_val",
            "roc_auc_test",
            "f1_test",
            "main_validation_metric",
            "main_test_metric",
            "accuracy_test",
        ]:
            if key in row and pd.notna(row[key]):
                try:
                    metrics[key] = float(row[key])
                except Exception:
                    metrics[key] = row[key]
        return metrics

    def list_available_models(self) -> dict:
        paths = self.expected_model_paths()
        metadata_paths = self._expected_metadata_paths()
        result: dict = {"models": {}, "missing": []}
        player_selected_path, selection_source = self._select_player_points_model_path()
        match_selected_path, match_selection_source = self._select_match_winner_model_path()
        for model_key, model_path in paths.items():
            if model_key == "player_points_model" and player_selected_path is not None:
                model_path = str(player_selected_path)
            if model_key == "match_winner_model" and match_selected_path is not None:
                model_path = str(match_selected_path)
            exists = Path(model_path).exists()
            md_path = metadata_paths[model_key]
            if model_key == "player_points_model" and player_selected_path is not None:
                md_path = str(self._metadata_path_for_model_path(player_selected_path))
            if model_key == "match_winner_model" and match_selected_path is not None:
                md_path = str(self._metadata_path_for_model_path(match_selected_path))
            metadata_exists = Path(md_path).exists()
            result["models"][model_key] = {
                "path": model_path,
                "exists": exists,
                "metadata_path": md_path,
                "metadata_exists": metadata_exists,
            }
            if model_key == "player_points_model":
                result["models"][model_key]["selection_source"] = selection_source
            if model_key == "match_winner_model":
                result["models"][model_key]["selection_source"] = match_selection_source
            if not exists:
                result["missing"].append(model_path)
        return result

    def load_player_points_model(self):
        model_path, _ = self._select_player_points_model_path()
        if model_path is None:
            return None, "Player points model not found in artifacts and PLAYER_POINTS_MODEL_PATH."
        try:
            return load_model_safely(str(model_path), require_predict_proba=False), None
        except ModelLoadError as exc:
            return None, str(exc)

    def load_player_points_model_bundle(self) -> dict:
        model_path, selection_source = self._select_player_points_model_path()
        if model_path is None:
            return {
                "status": "model_unavailable",
                "error": "Player points model not found.",
                "selection_source": selection_source,
            }

        model, model_error = self.load_player_points_model()
        if model is None:
            return {
                "status": "model_unavailable",
                "error": model_error,
                "model_path": str(model_path),
                "selection_source": selection_source,
            }

        metadata = None
        metadata_error = None
        metadata_path = self._metadata_path_for_model_path(model_path)
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                metadata_error = str(exc)
        else:
            metadata_error = f"Metadata file not found: {metadata_path}"

        model_name = None
        if metadata and metadata.get("model_name"):
            model_name = str(metadata["model_name"])
        else:
            model_name = model_path.stem

        metrics = {}
        if metadata and isinstance(metadata.get("metrics"), dict):
            metrics = metadata["metrics"]
        else:
            metrics = self._metrics_for_player_model_name(model_name=model_name)

        feature_list, feature_error = self.load_model_feature_list("player_points_model")
        return {
            "status": "ok",
            "model": model,
            "model_path": str(model_path),
            "model_name": model_name,
            "model_type": type(model).__name__,
            "model_metadata_path": str(metadata_path),
            "model_metadata_available": bool(metadata is not None),
            "selection_source": selection_source,
            "metadata": metadata,
            "metadata_error": metadata_error,
            "metrics": metrics,
            "feature_list": feature_list or [],
            "feature_error": feature_error,
        }

    def load_match_winner_model(self):
        model_path, _ = self._select_match_winner_model_path()
        if model_path is None:
            return None, "Match winner model not found in artifacts and MATCH_WINNER_MODEL_PATH."
        try:
            return load_model_safely(str(model_path), require_predict_proba=False), None
        except ModelLoadError as exc:
            return None, str(exc)

    def load_match_winner_model_bundle(self) -> dict:
        model_path, selection_source = self._select_match_winner_model_path()
        if model_path is None:
            return {
                "status": "model_unavailable",
                "error": "Match winner model not found.",
                "selection_source": selection_source,
            }

        model, model_error = self.load_match_winner_model()
        if model is None:
            return {
                "status": "model_unavailable",
                "error": model_error,
                "model_path": str(model_path),
                "selection_source": selection_source,
            }

        metadata = None
        metadata_error = None
        metadata_path = self._metadata_path_for_model_path(model_path)
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                metadata_error = str(exc)
        else:
            metadata_error = f"Metadata file not found: {metadata_path}"

        model_name = str(metadata.get("model_name")) if metadata and metadata.get("model_name") else model_path.stem
        metrics = metadata.get("metrics", {}) if metadata and isinstance(metadata.get("metrics"), dict) else self._metrics_for_match_model_name(model_name)
        feature_list, feature_error = self.load_model_feature_list("match_winner_model")
        return {
            "status": "ok",
            "model": model,
            "model_path": str(model_path),
            "model_name": model_name,
            "model_type": type(model).__name__,
            "model_metadata_path": str(metadata_path),
            "model_metadata_available": bool(metadata is not None),
            "selection_source": selection_source,
            "metadata": metadata,
            "metadata_error": metadata_error,
            "metrics": metrics,
            "feature_list": feature_list or [],
            "feature_error": feature_error,
        }

    def load_model_metadata(self, model_key: str) -> tuple[dict | None, str | None]:
        metadata_paths = self._expected_metadata_paths()
        if model_key not in metadata_paths:
            return None, f"Unsupported model key: {model_key}"
        path = Path(metadata_paths[model_key])
        if model_key == "player_points_model":
            selected_path, _ = self._select_player_points_model_path()
            if selected_path is not None:
                path = self._metadata_path_for_model_path(selected_path)
        if model_key == "match_winner_model":
            selected_path, _ = self._select_match_winner_model_path()
            if selected_path is not None:
                path = self._metadata_path_for_model_path(selected_path)
        if not path.exists():
            return None, f"Metadata file not found: {path}"
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), None
        except Exception as exc:
            return None, f"Failed to load metadata {path.name}: {exc}"

    def load_model_feature_list(self, model_key: str) -> tuple[list[str] | None, str | None]:
        metadata, metadata_error = self.load_model_metadata(model_key)
        if metadata:
            if isinstance(metadata.get("features"), list):
                return metadata["features"], None
            if isinstance(metadata.get("feature_list"), list):
                return metadata["feature_list"], None

        model = None
        model_error = None
        if model_key == "player_points_model":
            model, model_error = self.load_player_points_model()
        elif model_key == "match_winner_model":
            model, model_error = self.load_match_winner_model()
        else:
            return None, f"Unsupported model key: {model_key}"

        if model is not None and hasattr(model, "feature_names_in_"):
            return [str(x) for x in list(model.feature_names_in_)], None

        if metadata_error and model_error:
            return None, f"{metadata_error}; {model_error}"
        if metadata_error:
            return None, metadata_error
        return None, model_error or "Feature list not available."
