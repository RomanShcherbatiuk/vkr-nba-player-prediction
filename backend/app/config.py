import os
from pathlib import Path


_fallback_root = Path.cwd()
_module_path = Path(__file__).resolve()
if len(_module_path.parents) > 3:
    _fallback_root = _module_path.parents[3]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(_fallback_root)))

MATCH_DATASET_PATH = os.getenv(
    "MATCH_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "match_prediction_dataset.csv"),
)
MATCH_FEATURE_SETS_PATH = os.getenv(
    "MATCH_FEATURE_SETS_PATH",
    str(PROJECT_ROOT / "data" / "match_prediction_feature_sets.json"),
)
PLAYER_POINTS_DATASET_PATH = os.getenv(
    "PLAYER_POINTS_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "player_points_prediction_dataset.csv"),
)
PLAYER_POINTS_FEATURE_SETS_PATH = os.getenv(
    "PLAYER_POINTS_FEATURE_SETS_PATH",
    str(PROJECT_ROOT / "data" / "player_points_feature_sets.json"),
)
MODEL_ARTIFACTS_DIR = os.getenv(
    "MODEL_ARTIFACTS_DIR",
    str(PROJECT_ROOT / "models"),
)
MATCH_WINNER_MODEL_PATH = os.getenv(
    "MATCH_WINNER_MODEL_PATH",
    str(Path(MODEL_ARTIFACTS_DIR) / "match_winner_model.joblib"),
)
MATCH_WINNER_MODEL_METADATA_PATH = os.getenv(
    "MATCH_WINNER_MODEL_METADATA_PATH",
    str(Path(MODEL_ARTIFACTS_DIR) / "match_winner_model_metadata.json"),
)
PLAYER_POINTS_MODEL_PATH = os.getenv(
    "PLAYER_POINTS_MODEL_PATH",
    str(Path(MODEL_ARTIFACTS_DIR) / "player_points_model.joblib"),
)
PLAYER_POINTS_MODEL_METADATA_PATH = os.getenv(
    "PLAYER_POINTS_MODEL_METADATA_PATH",
    str(Path(MODEL_ARTIFACTS_DIR) / "player_points_model_metadata.json"),
)
REPORTS_DIR = os.getenv(
    "REPORTS_DIR",
    "",
)

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
