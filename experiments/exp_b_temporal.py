"""
HeatWise Experiment B: Temporal & Episode-Grouped Validation
Compares random row split vs. episode-grouped split vs. temporal block split
to rigorously quantify data leakage and out-of-trajectory generalization.
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
from experiments.dataset_utils import generate_sequential_dataset, split_episode_grouped, split_temporal_blocks
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_experiment_b(dataset_path: str = None) -> List[Dict[str, Any]]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
    else:
        cached_a2 = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
        if os.path.exists(cached_a2):
            print(f"Loading cached A2 dataset from {cached_a2}...")
            df = pd.read_csv(cached_a2)
        else:
            print("Generating 10,000-sample dataset for temporal validation...")
            df = generate_sequential_dataset(n_samples=10000, seed=config.TELEMETRY_RANDOM_SEED)

    results = []

    # 1. Random Row Split (Control)
    print("\n--- [EXP B] Protocol 1: Random Row Split (80/20) ---")
    X = df[ml_model.FEATURE_COLUMNS]
    y = df[ml_model.TARGET_COLUMN]
    X_tr_rnd, X_te_rnd, y_tr_rnd, y_te_rnd = train_test_split(
        X, y, test_size=0.20, random_state=config.ML_RANDOM_SEED
    )
    m_rnd = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    t0 = time.time()
    m_rnd.fit(X_tr_rnd, y_tr_rnd)
    tr_time_rnd = round(time.time() - t0, 3)
    y_pred_rnd = m_rnd.predict(X_te_rnd)
    s_mae_rnd = round(float(mean_absolute_error(y_te_rnd, y_pred_rnd)), 3)
    s_rmse_rnd = round(float(np.sqrt(mean_squared_error(y_te_rnd, y_pred_rnd))), 3)
    s_r2_rnd = round(float(r2_score(y_te_rnd, y_pred_rnd)), 4)
    dyn_rnd = run_dynamic_closed_loop_eval(m_rnd, n_decisions_per_scenario=50, seed=42)

    results.append({
        "split_method": "Random Row Split (80/20)",
        "train_samples": len(X_tr_rnd),
        "test_samples": len(X_te_rnd),
        "train_time_s": tr_time_rnd,
        "static_mae": s_mae_rnd,
        "static_rmse": s_rmse_rnd,
        "static_r2": s_r2_rnd,
        "dynamic_mae": dyn_rnd["dynamic_mae"],
        "dynamic_top1_pct": dyn_rnd["top1_agreement_pct"],
        "dynamic_top3_pct": dyn_rnd["top3_overlap_pct"],
        "spearman_rho": dyn_rnd["spearman_rho"],
        "optimality_gap": dyn_rnd["optimality_gap"],
        "materially_worse_rate_pct": dyn_rnd["materially_worse_rate_pct"],
        "leakage_risk": "High (intra-trajectory rows in train & test)"
    })

    # 2. Episode-Grouped Split (Out-of-Trajectory)
    print("\n--- [EXP B] Protocol 2: Episode-Grouped Split (Out-of-Trajectory) ---")
    tr_df_grp, te_df_grp = split_episode_grouped(df, test_size=0.20, seed=config.ML_RANDOM_SEED)
    X_tr_grp = tr_df_grp[ml_model.FEATURE_COLUMNS]
    y_tr_grp = tr_df_grp[ml_model.TARGET_COLUMN]
    X_te_grp = te_df_grp[ml_model.FEATURE_COLUMNS]
    y_te_grp = te_df_grp[ml_model.TARGET_COLUMN]

    m_grp = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    t0 = time.time()
    m_grp.fit(X_tr_grp, y_tr_grp)
    tr_time_grp = round(time.time() - t0, 3)
    y_pred_grp = m_grp.predict(X_te_grp)
    s_mae_grp = round(float(mean_absolute_error(y_te_grp, y_pred_grp)), 3)
    s_rmse_grp = round(float(np.sqrt(mean_squared_error(y_te_grp, y_pred_grp))), 3)
    s_r2_grp = round(float(r2_score(y_te_grp, y_pred_grp)), 4)
    dyn_grp = run_dynamic_closed_loop_eval(m_grp, n_decisions_per_scenario=50, seed=42)

    results.append({
        "split_method": "Episode-Grouped Split",
        "train_samples": len(X_tr_grp),
        "test_samples": len(X_te_grp),
        "train_time_s": tr_time_grp,
        "static_mae": s_mae_grp,
        "static_rmse": s_rmse_grp,
        "static_r2": s_r2_grp,
        "dynamic_mae": dyn_grp["dynamic_mae"],
        "dynamic_top1_pct": dyn_grp["top1_agreement_pct"],
        "dynamic_top3_pct": dyn_grp["top3_overlap_pct"],
        "spearman_rho": dyn_grp["spearman_rho"],
        "optimality_gap": dyn_grp["optimality_gap"],
        "materially_worse_rate_pct": dyn_grp["materially_worse_rate_pct"],
        "leakage_risk": "Zero (strictly disjoint trajectories)"
    })

    # 3. Temporal Block Split (Early steps vs Late steps)
    print("\n--- [EXP B] Protocol 3: Temporal Block Split (Step <=14 vs Step >14) ---")
    tr_df_tmp, te_df_tmp = split_temporal_blocks(df, train_step_cutoff=14)
    X_tr_tmp = tr_df_tmp[ml_model.FEATURE_COLUMNS]
    y_tr_tmp = tr_df_tmp[ml_model.TARGET_COLUMN]
    X_te_tmp = te_df_tmp[ml_model.FEATURE_COLUMNS]
    y_te_tmp = te_df_tmp[ml_model.TARGET_COLUMN]

    m_tmp = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    t0 = time.time()
    m_tmp.fit(X_tr_tmp, y_tr_tmp)
    tr_time_tmp = round(time.time() - t0, 3)
    y_pred_tmp = m_tmp.predict(X_te_tmp)
    s_mae_tmp = round(float(mean_absolute_error(y_te_tmp, y_pred_tmp)), 3)
    s_rmse_tmp = round(float(np.sqrt(mean_squared_error(y_te_tmp, y_pred_tmp))), 3)
    s_r2_tmp = round(float(r2_score(y_te_tmp, y_pred_tmp)), 4)
    dyn_tmp = run_dynamic_closed_loop_eval(m_tmp, n_decisions_per_scenario=50, seed=42)

    results.append({
        "split_method": "Temporal Block Split (Forward-Horizon)",
        "train_samples": len(X_tr_tmp),
        "test_samples": len(X_te_tmp),
        "train_time_s": tr_time_tmp,
        "static_mae": s_mae_tmp,
        "static_rmse": s_rmse_tmp,
        "static_r2": s_r2_tmp,
        "dynamic_mae": dyn_tmp["dynamic_mae"],
        "dynamic_top1_pct": dyn_tmp["top1_agreement_pct"],
        "dynamic_top3_pct": dyn_tmp["top3_overlap_pct"],
        "spearman_rho": dyn_tmp["spearman_rho"],
        "optimality_gap": dyn_tmp["optimality_gap"],
        "materially_worse_rate_pct": dyn_tmp["materially_worse_rate_pct"],
        "leakage_risk": "Zero (train on historical past, test on sustained future)"
    })

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "temporal_validation.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "temporal_validation.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment_b()
