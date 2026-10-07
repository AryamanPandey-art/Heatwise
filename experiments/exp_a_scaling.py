"""
HeatWise Experiment A: Dataset Scaling
Evaluates impact of scaling sequential transition telemetry from 3,500 to 10,000 and 20,000 samples.
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import ml_model
from experiments.dataset_utils import generate_sequential_dataset
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_experiment_a() -> List[Dict[str, Any]]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = [
        {"name": "A1_3500", "n_samples": 3500},
        {"name": "A2_10000", "n_samples": 10000},
        {"name": "A3_20000", "n_samples": 20000}
    ]

    results = []

    for cfg in configs:
        name = cfg["name"]
        n_samples = cfg["n_samples"]
        print(f"\n--- [EXP A] Generating Dataset: {name} (N={n_samples}) ---")
        t0 = time.time()
        df = generate_sequential_dataset(n_samples=n_samples, seed=config.TELEMETRY_RANDOM_SEED)
        gen_time = round(time.time() - t0, 3)

        # Save dataset artifact
        csv_path = os.path.join(RESULTS_DIR, f"dataset_{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved dataset to {csv_path} (Gen time: {gen_time}s)")

        episode_count = int(df["episode_id"].nunique())
        scenario_dist = df["scenario_name"].value_counts().to_dict()

        # Static train/test split (80/20 standard)
        X = df[ml_model.FEATURE_COLUMNS]
        y = df[ml_model.TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=config.ML_RANDOM_SEED
        )

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=config.ML_RANDOM_SEED,
            n_jobs=-1
        )

        t_train0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t_train0, 3)

        y_pred = model.predict(X_test)
        s_mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
        s_rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
        s_r2 = round(float(r2_score(y_test, y_pred)), 4)

        print(f"Static Fit: MAE={s_mae}°C, RMSE={s_rmse}°C, R²={s_r2*100:.2f}% (Train time: {train_time}s)")

        # Dynamic closed-loop evaluation (250 decisions)
        print("Running dynamic closed-loop evaluation (250 decisions)...")
        dyn_res = run_dynamic_closed_loop_eval(model, n_decisions_per_scenario=50, seed=42)
        print(f"Dynamic Res: MAE={dyn_res['dynamic_mae']}°C, Top-1={dyn_res['top1_agreement_pct']}%, Top-3={dyn_res['top3_overlap_pct']}%, OptGap={dyn_res['optimality_gap']}°C")

        res_entry = {
            "experiment": "Dataset Scaling",
            "config_id": name,
            "sample_count": n_samples,
            "episode_count": episode_count,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "scenario_distribution": scenario_dist,
            "data_gen_time_s": gen_time,
            "train_time_s": train_time,
            "static_mae": s_mae,
            "static_rmse": s_rmse,
            "static_r2": s_r2,
            "dynamic_mae": dyn_res["dynamic_mae"],
            "dynamic_top1_pct": dyn_res["top1_agreement_pct"],
            "dynamic_top3_pct": dyn_res["top3_overlap_pct"],
            "spearman_rho": dyn_res["spearman_rho"],
            "optimality_gap": dyn_res["optimality_gap"],
            "materially_worse_rate_pct": dyn_res["materially_worse_rate_pct"],
            "unsafe_placements": dyn_res["unsafe_placements_count"]
        }
        results.append(res_entry)

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "dataset_scaling.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "dataset_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment_a()
