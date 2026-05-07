from pathlib import Path

import pandas as pd

from app import config


class ArtifactRepository:
    def __init__(self) -> None:
        self.reports_dir = Path(config.REPORTS_DIR) if config.REPORTS_DIR else None
        self.figures_dir = self.reports_dir.parent / "figures" if self.reports_dir else None

    def get_artifact_dirs(self) -> dict:
        return {
            "model_artifacts_dir": config.MODEL_ARTIFACTS_DIR,
            "reports_dir": config.REPORTS_DIR,
        }

    def load_baseline_results(self) -> tuple[pd.DataFrame | None, str | None]:
        if self.reports_dir is None:
            return None, "Baseline summary is not configured."
        path = self.reports_dir / "baseline_summary_all_tasks.csv"
        if not path.exists():
            return None, f"Baseline summary not found: {path}"
        try:
            return pd.read_csv(path), None
        except Exception as exc:
            return None, f"Failed to load baseline summary {path.name}: {exc}"

    def load_model_metrics(self) -> dict:
        if self.reports_dir is None:
            return {"loaded": {}, "missing": {}, "errors": {}}
        metric_files = [
            "baseline_summary_all_tasks.csv",
            "match_classification_results.csv",
            "match_tuning_results.csv",
            "player_points_regression_results.csv",
            "player_points_tuning_results.csv",
        ]
        output: dict = {"loaded": {}, "missing": {}, "errors": {}}
        for file_name in metric_files:
            path = self.reports_dir / file_name
            if not path.exists():
                output["missing"][file_name] = str(path)
                continue
            try:
                df = pd.read_csv(path)
                output["loaded"][file_name] = {
                    "path": str(path),
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                }
            except Exception as exc:
                output["errors"][file_name] = str(exc)
        return output

    def load_available_reports(self) -> list[dict]:
        if self.reports_dir is None or not self.reports_dir.exists():
            return []
        reports: list[dict] = []
        for path in sorted(self.reports_dir.glob("*")):
            if path.is_file():
                reports.append(
                    {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
                )
        return reports

    def load_available_figures(self) -> list[dict]:
        if self.figures_dir is None or not self.figures_dir.exists():
            return []
        figures: list[dict] = []
        for path in sorted(self.figures_dir.glob("*")):
            if path.is_file():
                figures.append(
                    {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
                )
        return figures
