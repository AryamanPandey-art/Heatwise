"""
HeatWise Experiment C: Leave-One-Scenario-Out Holdout Validation
Trains on 4 operational scenarios and evaluates strictly out-of-distribution
on the held-out 5th scenario to quantify robustness to unseen failure regimes.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import ml_model
from experiments.dataset_utils import generate_sequential_dataset, SCENARIOS
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_experiment_c(dataset_path: str = None) -> List[Dict[str, Any]]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if dataset_path and os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        cached_a2 = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
        if os.path.exists(cached_a2):
            df = pd.read_csv(cached_a2)
        else:
            print("Generating balanced sequential dataset for scenario holdout...")
            df = generate_sequential_dataset(n_samples=10000, seed=config.TELEMETRY_RANDOM_SEED)

    results = []

    for held_out in SCENARIOS:
        print(f"\n--- [EXP C] Holding out scenario: {held_out} ---")
        train_df = df[df["scenario_name"] != held_out]
        test_df = df[df["scenario_name"] == held_out]

        X_train = train_df[ml_model.FEATURE_COLUMNS]
        y_train = train_df[ml_model.TARGET_COLUMN]
        X_test = test_df[ml_model.FEATURE_COLUMNS]
        y_test = test_df[ml_model.TARGET_COLUMN]

        model = RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
            random_state=config.ML_RANDOM_SEED, n_jobs=-1
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 3)

        y_pred = model.predict(X_test)
        residuals = y_pred - y_test.values

        s_mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
        s_rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
        s_r2 = round(float(r2_score(y_test, y_pred)), 4)
        res_mean = round(float(np.mean(residuals)), 3)
        res_std = round(float(np.std(residuals)), 3)
        res_p95 = round(float(np.percentile(np.abs(residuals), 95)), 3)

        # Dynamic closed-loop placement strictly under held-out scenario
        dyn_eval = run_dynamic_closed_loop_eval(
            model,
            n_decisions_per_scenario=50,
            seed=42,
            target_scenarios=[held_out]
        )

        entry = {
            "held_out_scenario": held_out,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "fit_time_s": fit_time,
            "static_mae": s_mae,
            "static_rmse": s_rmse,
            "static_r2": s_r2,
            "residual_mean": res_mean,
            "residual_std": res_std,
            "residual_p95_abs": res_p95,
            "dynamic_mae": dyn_eval["dynamic_mae"],
            "top1_agreement_pct": dyn_eval["top1_agreement_pct"],
            "top3_overlap_pct": dyn_eval["top3_overlap_pct"],
            "spearman_rho": dyn_eval["spearman_rho"],
            "optimality_gap": dyn_eval["optimality_gap"],
            "materially_worse_rate_pct": dyn_eval["materially_worse_rate_pct"]
        }
        results.append(entry)
        print(f"Results for {held_out}: MAE={s_mae}°C, R²={s_r2*100:.2f}%, Top-1={dyn_eval['top1_agreement_pct']}%, OptGap={dyn_eval['optimality_gap']}°C")

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "scenario_holdout.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "scenario_holdout.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment_c()
