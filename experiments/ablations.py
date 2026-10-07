"""
HeatWise Experiment G: Feature & Heuristic Ablation Study
Deconstructs spatial/cooling feature contributions (G1-G5) and compares
temperature-only heuristic (G6) against multi-factor scoring (G7).
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
import heatwise
from experiments.dataset_utils import generate_sequential_dataset
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_experiment_ablations(dataset_path: str = None) -> List[Dict[str, Any]]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if dataset_path and os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        cached_a2 = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
        if os.path.exists(cached_a2):
            df = pd.read_csv(cached_a2)
        else:
            df = generate_sequential_dataset(n_samples=10000, seed=config.TELEMETRY_RANDOM_SEED)

    full_features = list(ml_model.FEATURE_COLUMNS)

    # Feature ablation definitions
    feature_variants = [
        ("G1_Full_17_Features", full_features),
        ("G2_No_Neighbor_Temp", [f for f in full_features if f != "neighbor_temperature"]),
        ("G3_No_Cooling_Capacity", [f for f in full_features if f != "cooling_capacity"]),
        ("G4_No_Recirculation", [f for f in full_features if f != "recirculation_factor"]),
        ("G5_No_Neighbor_Signals", [f for f in full_features if f not in ("neighbor_temperature", "neighbor_utilization")]),
    ]

    results = []

    # Train and evaluate feature variants (using canonical heuristic)
    for name, feats in feature_variants:
        print(f"\n--- [EXP G] Evaluating Feature Variant: {name} (N={len(feats)} features) ---")
        X = df[feats]
        y = df[ml_model.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=config.ML_RANDOM_SEED
        )

        model = RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
            random_state=config.ML_RANDOM_SEED, n_jobs=-1
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 3)

        y_pred = model.predict(X_test)
        s_mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
        s_rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
        s_r2 = round(float(r2_score(y_test, y_pred)), 4)

        dyn_res = run_dynamic_closed_loop_eval(
            model,
            n_decisions_per_scenario=50,
            seed=42,
            feature_columns=feats
        )

        entry = {
            "variant_id": name,
            "ablation_type": "Feature Ablation",
            "feature_count": len(feats),
            "heuristic_type": "Current HeatWise Multi-Factor",
            "fit_time_s": fit_time,
            "static_mae": s_mae,
            "static_rmse": s_rmse,
            "static_r2": s_r2,
            "dynamic_mae": dyn_res["dynamic_mae"],
            "top1_agreement_pct": dyn_res["top1_agreement_pct"],
            "top3_overlap_pct": dyn_res["top3_overlap_pct"],
            "spearman_rho": dyn_res["spearman_rho"],
            "optimality_gap": dyn_res["optimality_gap"],
            "materially_worse_rate_pct": dyn_res["materially_worse_rate_pct"],
            "unsafe_placements": dyn_res["unsafe_placements_count"]
        }
        results.append(entry)
        print(f"{name}: Static MAE={s_mae}°C, Dynamic MAE={dyn_res['dynamic_mae']}°C, Top-1={dyn_res['top1_agreement_pct']}%, OptGap={dyn_res['optimality_gap']}°C")

    # Heuristic Ablations using Full 17-feature model
    print("\n--- [EXP G] Evaluating Heuristic Ablations (G6 vs G7) ---")
    base_model = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    X_full = df[full_features]
    y_full = df[ml_model.TARGET_COLUMN]
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.20, random_state=config.ML_RANDOM_SEED)
    base_model.fit(X_tr, y_tr)

    # G6: Temperature-Only Heuristic (100% weight on temperature safety / minimal predicted temp)
    def temp_only_scoring(rack, pred_temp, has_gpu, has_mem):
        is_hotspot = pred_temp >= config.TEMP_CRITICAL
        # Score is purely inverted predicted temperature (higher score = lower temperature)
        score = max(0.0, 100.0 - pred_temp)
        if is_hotspot:
            score -= 45.0 + (pred_temp - config.TEMP_CRITICAL) * 5.0
        if not has_gpu:
            score -= 50.0
        if not has_mem:
            score -= 50.0
        is_rejected = (not has_gpu) or (not has_mem) or is_hotspot
        return float(np.clip(score, 0.0, 100.0)), is_rejected

    dyn_g6 = run_dynamic_closed_loop_eval(
        base_model,
        n_decisions_per_scenario=50,
        seed=42,
        custom_scoring_fn=temp_only_scoring
    )

    results.append({
        "variant_id": "G6_Temp_Only_Heuristic",
        "ablation_type": "Heuristic Ablation",
        "feature_count": 17,
        "heuristic_type": "Pure Temperature Minimization (100% Temp)",
        "fit_time_s": results[0]["fit_time_s"],
        "static_mae": results[0]["static_mae"],
        "static_rmse": results[0]["static_rmse"],
        "static_r2": results[0]["static_r2"],
        "dynamic_mae": dyn_g6["dynamic_mae"],
        "top1_agreement_pct": dyn_g6["top1_agreement_pct"],
        "top3_overlap_pct": dyn_g6["top3_overlap_pct"],
        "spearman_rho": dyn_g6["spearman_rho"],
        "optimality_gap": dyn_g6["optimality_gap"],
        "materially_worse_rate_pct": dyn_g6["materially_worse_rate_pct"],
        "unsafe_placements": dyn_g6["unsafe_placements_count"]
    })
    print(f"G6 Temp-Only: Dynamic MAE={dyn_g6['dynamic_mae']}°C, Top-1={dyn_g6['top1_agreement_pct']}%, OptGap={dyn_g6['optimality_gap']}°C")

    # G7: Current HeatWise Full Multi-Factor Heuristic
    results.append({
        "variant_id": "G7_Full_HeatWise_Heuristic",
        "ablation_type": "Heuristic Ablation",
        "feature_count": 17,
        "heuristic_type": "Multi-Factor (Safety 35%, Headroom 25%, GPU 20%, Cool 10%, Nbr 10%)",
        "fit_time_s": results[0]["fit_time_s"],
        "static_mae": results[0]["static_mae"],
        "static_rmse": results[0]["static_rmse"],
        "static_r2": results[0]["static_r2"],
        "dynamic_mae": results[0]["dynamic_mae"],
        "top1_agreement_pct": results[0]["top1_agreement_pct"],
        "top3_overlap_pct": results[0]["top3_overlap_pct"],
        "spearman_rho": results[0]["spearman_rho"],
        "optimality_gap": results[0]["optimality_gap"],
        "materially_worse_rate_pct": results[0]["materially_worse_rate_pct"],
        "unsafe_placements": results[0]["unsafe_placements"]
    })
    print(f"G7 Full Heuristic: Dynamic MAE={results[0]['dynamic_mae']}°C, Top-1={results[0]['top1_agreement_pct']}%, OptGap={results[0]['optimality_gap']}°C")

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "ablations.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "ablations.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment_ablations()
