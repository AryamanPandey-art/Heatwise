"""
HeatWise Staging: Thermal Predictor Wrapper
Provides a staging drop-in replacement for ThermalMLPredictor,
loading candidate_production_model_10k.joblib with n_jobs=1.
"""

import os
import sys
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config
import ml_model

STAGING_DIR = os.path.dirname(os.path.abspath(__file__))


class StagingThermalPredictor:
    """
    Staging candidate predictor matching the ThermalMLPredictor contract.
    Loads candidate_production_model_10k.joblib with n_jobs=1.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(STAGING_DIR, "candidate_production_model_10k.joblib")
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_importances: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
        self.test_eval_df: pd.DataFrame = pd.DataFrame()

        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            # Ensure n_jobs is 1 for serial execution
            self.model.n_jobs = 1
            self.is_trained = True

            importances = self.model.feature_importances_
            self.feature_importances = {
                ml_model.FEATURE_LABELS[col]: round(float(imp) * 100.0, 2)
                for col, imp in sorted(zip(ml_model.FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
            }
            self.metrics = {
                "algorithm": "Random Forest Regressor (120 Trees, n_jobs=1, 10k Telemetry)",
                "total_samples": 10000,
                "n_estimators": 120,
                "max_depth": 14,
                "n_jobs": 1
            }
        else:
            raise FileNotFoundError(f"Candidate model artifact not found at: {self.model_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[ml_model.FEATURE_COLUMNS])

    def predict_rack_placement(self, rack: Dict[str, Any], workload) -> Tuple[float, float]:
        """
        Exact match for ThermalMLPredictor.predict_rack_placement.
        """
        input_df = ml_model.build_prediction_features(rack, workload)
        pred_temp = round(float(self.model.predict(input_df)[0]), 1)
        temp_delta = round(pred_temp - rack["temperature"], 1)
        return pred_temp, temp_delta


_staging_instance = None


def get_staging_predictor() -> StagingThermalPredictor:
    """Returns singleton of the staging predictor."""
    global _staging_instance
    if _staging_instance is None:
        _staging_instance = StagingThermalPredictor()
    return _staging_instance
