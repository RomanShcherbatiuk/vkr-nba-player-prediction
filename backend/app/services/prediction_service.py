class PredictionService:
    def predict(self) -> dict:
        return {
            "status": "model_unavailable",
            "model_mode": "real",
            "message": "Emergency fallback: concrete prediction service endpoint is unavailable.",
        }
