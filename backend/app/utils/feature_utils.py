import pandas as pd


def normalize_feature_names(feature_names: list[str]) -> list[str]:
    return [name.strip() for name in feature_names if isinstance(name, str) and name.strip()]


def check_feature_availability(df: pd.DataFrame, feature_list: list[str]) -> dict:
    normalized = normalize_feature_names(feature_list)
    available = [feature for feature in normalized if feature in df.columns]
    missing = [feature for feature in normalized if feature not in df.columns]
    return {
        "expected_count": len(normalized),
        "available_count": len(available),
        "missing_count": len(missing),
        "available_features": available,
        "missing_features": missing,
    }


def align_dataframe_to_model_features(
    df: pd.DataFrame,
    feature_list: list[str],
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    normalized = normalize_feature_names(feature_list)
    report = check_feature_availability(df, normalized)

    aligned = df.copy()
    for feature in report["missing_features"]:
        aligned[feature] = fill_value

    aligned = aligned.reindex(columns=normalized)
    aligned = aligned.fillna(fill_value)
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(fill_value)

    report["fill_value"] = fill_value
    return aligned, report


def build_missing_features_report(feature_check: dict) -> dict:
    return {
        "status": "ok" if feature_check["missing_count"] == 0 else "missing_features",
        "expected_count": feature_check["expected_count"],
        "available_count": feature_check["available_count"],
        "missing_count": feature_check["missing_count"],
        "missing_features": feature_check["missing_features"],
    }
