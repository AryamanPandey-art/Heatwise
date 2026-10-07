"""
HeatWise Experiment D: ExtraTreesRegressor Comparison
Evaluates Extremely Randomized Trees against Random Forest baseline
under identical data splits, features, and dynamic placement oracle loops.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import ml_model
from experiments.dataset_utils import generate_sequential_dataset
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_experiment_d(dataset_path: str = None) -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if dataset_path and os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        cached_a2 = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
        if os.path.exists(cached_a2):
            df = pd.read_csv(cached_a2)
        else:
            df = generate_sequential_dataset(n_samples=10000, seed=config.TELEMETRY_RANDOM_SEED)

    X = df[ml_model.FEATURE_COLUMNS]
    y = df[ml_model.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.ML_RANDOM_SEED
    )

    print("\n--- [EXP D] Training ExtraTreesRegressor ---")
    model = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)

    # Benchmark single-sample inference time
    t_inf_start = time.time()
    for _ in range(500):
        _ = model.predict(X_test.iloc[0:1])
    inf_time_ms = round((time.time() - t_inf_start) / 500 * 1000.0, 3)

    y_pred = model.predict(X_test)
    s_mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
    s_rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
    s_r2 = round(float(r2_score(y_test, y_pred)), 4)

    print(f"ExtraTrees Fit: MAE={s_mae}°C, RMSE={s_rmse}°C, R²={s_r2*100:.2f}%, Inf={inf_time_ms}ms")

    print("Running dynamic closed-loop evaluation (250 decisions)...")
    dyn_res = run_dynamic_closed_loop_eval(model, n_decisions_per_scenario=50, seed=42)

    entry = {
        "model_name": "ExtraTreesRegressor (120 Trees)",
        "train_time_s": train_time,
        "single_rack_inference_ms": inf_time_ms,
        "static_mae": s_mae,
        "static_rmse": s_rmse,
        "static_r2": s_r2,
        "dynamic_mae": dyn_res["dynamic_mae"],
        "top1_agreement_pct": dyn_res["top1_agreement_pct"],
        "top3_overlap_pct": dyn_res["top3_overlap_pct"],
        "spearman_rho": dyn_res["spearman_rho"],
        "optimality_gap": dyn_res["optimality_gap"],
        "materially_worse_rate_pct": dyn_res["materially_worse_rate_pct"]
    }

    return entry


if __name__ == "__main__":
    res = run_experiment_d()
    print(json.dumps(res, indent=2))
