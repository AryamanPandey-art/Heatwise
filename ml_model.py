"""
HeatWise: Machine Learning Thermal Prediction Module
Implements a Random Forest Regressor trained on physics-grounded telemetry
to predict rack equilibrium temperature after workload placement.
Computes evaluation metrics (MAE, RMSE, R²) and feature importances.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import simulation

FEATURE_COLUMNS = [
    "current_temp",
    "cpu_utilization",
    "gpu_utilization",
    "power_watts",
    "cooling_capacity",
    "neighbor_avg_temp",
    "workload_gpu_demand",
    "workload_intensity_val",
    "workload_power_est"
]

FEATURE_LABELS = {
    "current_temp": "Current Temperature (°C)",
    "cpu_utilization": "CPU Utilization (%)",
    "gpu_utilization": "GPU Utilization (%)",
    "power_watts": "Rack Power (W)",
    "cooling_capacity": "Cooling Capacity (%)",
    "neighbor_avg_temp": "Neighbor Avg Temp (°C)",
    "workload_gpu_demand": "Workload GPU Demand (%)",
    "workload_intensity_val": "Workload Intensity (1-4)",
    "workload_power_est": "Estimated Added Power (W)"
}

TARGET_COLUMN = "post_workload_temp"


class ThermalMLPredictor:
    """Wrapper class for the Random Forest Thermal Prediction Model."""

    def __init__(self, n_estimators: int = 120, max_depth: int = 14, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.feature_importances: Dict[str, float] = {}
        self.test_eval_df: pd.DataFrame = pd.DataFrame()

    def train(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Trains the Random Forest model on simulated telemetry data."""
        if df is None:
            df = simulation.generate_synthetic_training_data(n_samples=3500)

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        importances = self.model.feature_importances_
        feat_imp = {
            FEATURE_LABELS[col]: round(float(imp) * 100.0, 2)
            for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        }

        # Store test evaluation sample for plotting
        eval_sample = pd.DataFrame({
            "Actual (°C)": np.round(y_test.values[:100], 2),
            "Predicted (°C)": np.round(y_pred[:100], 2),
            "Residual (°C)": np.round((y_pred[:100] - y_test.values[:100]), 2)
        })

        self.metrics = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "r2_pct": round(r2 * 100.0, 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(df),
            "algorithm": "Random Forest Regressor (120 Trees)"
        }
        self.feature_importances = feat_imp
        self.test_eval_df = eval_sample
        self.is_trained = True

        return self.metrics

    def predict_rack_placement(
        self,
        rack: Dict[str, Any],
        workload_gpu_pct: float,
        workload_intensity_val: int,
        workload_power_est: float
    ) -> Tuple[float, float]:
        """
        Predicts the post-workload temperature for a single rack using the ML model.
        Returns: (predicted_temperature, temperature_delta)
        """
        if not self.is_trained:
            self.train()

        input_data = pd.DataFrame([{
            "current_temp": rack["temperature"],
            "cpu_utilization": rack["avg_cpu_util"],
            "gpu_utilization": rack["avg_gpu_util"],
            "power_watts": rack["total_power_w"],
            "cooling_capacity": rack["cooling_capacity_pct"],
            "neighbor_avg_temp": rack["neighbor_avg_temp"],
            "workload_gpu_demand": workload_gpu_pct,
            "workload_intensity_val": workload_intensity_val,
            "workload_power_est": workload_power_est
        }])[FEATURE_COLUMNS]

        pred_temp = round(float(self.model.predict(input_data)[0]), 1)
        temp_delta = round(pred_temp - rack["temperature"], 1)
        return pred_temp, temp_delta


# Global instance
_predictor_instance = None


def get_trained_predictor() -> ThermalMLPredictor:
    """Returns a singleton instance of the trained thermal ML predictor."""
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_trained:
        _predictor_instance = ThermalMLPredictor()
        _predictor_instance.train()
    return _predictor_instance
