from pathlib import Path
import pickle

import joblib


class ModelLoadError(Exception):
    pass


def _ensure_predict_interface(model, require_predict_proba: bool = False) -> None:
    if not hasattr(model, "predict"):
        raise ModelLoadError("Loaded object does not implement predict().")
    if require_predict_proba and not hasattr(model, "predict_proba"):
        raise ModelLoadError("Loaded model does not implement predict_proba().")


def load_model_safely(model_path: str, require_predict_proba: bool = False):
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    if path.suffix not in {".joblib", ".pkl"}:
        raise ModelLoadError(
            f"Unsupported model format for {path.name}. Expected .joblib or .pkl."
        )

    try:
        if path.suffix == ".joblib":
            model = joblib.load(path)
        else:
            with path.open("rb") as f:
                model = pickle.load(f)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model {path.name}: {exc}") from exc

    _ensure_predict_interface(model, require_predict_proba=require_predict_proba)
    return model
