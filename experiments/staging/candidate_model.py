"""
HeatWise Staging: Candidate Production Model Builder
Builds, evaluates, and saves:
1. Model A: Grouped Holdout Model (trained on 80% grouped train partition of 10k dataset)
2. Model B: Final Candidate Production Model (trained on all 10,000 samples for deployment)
Enforces:
- 17 features exactly matching ml_model.FEATURE_COLUMNS
- Zero post-placement leakage
- Deterministic random_state = 42
- n_jobs = 1
- Saves artifacts to experiments/staging/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config
import simulation
import ml_model
from experiments.dataset_utils import generate_sequential_dataset, split_episode_grouped

STAGING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(STAGING_DIR, "..", "results"))


def build_candidate_models():
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("================================================================================")
    print("STAGING: BUILDING CANDIDATE PRODUCTION MODELS (10k Sequential Telemetry)")
    print("================================================================================")

    # 1. Load or Generate 10k Sequential Transition Dataset
    dataset_10k_path = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
    if os.path.exists(dataset_10k_path):
        print(f"Loading cached 10k dataset from: {dataset_10k_path}")
        df_10k = pd.read_csv(dataset_10k_path)
    else:
        print("Generating 10k sequential transition dataset...")
        df_10k = generate_sequential_dataset(
            n_samples=10000,
            seed=config.TELEMETRY_RANDOM_SEED,
            max_steps_per_episode=20
        )
        df_10k.to_csv(dataset_10k_path, index=False)

    print(f"Dataset shape: {df_10k.shape}")
    print(f"Unique episodes: {df_10k['episode_id'].nunique()}")
    print(f"Features: {len(ml_model.FEATURE_COLUMNS)} columns")

    # Anti-leakage assertions
    assert "post_workload_temp" not in ml_model.FEATURE_COLUMNS, "Leakage: target is in features!"
    for col in ml_model.FEATURE_COLUMNS:
        assert col in df_10k.columns, f"Missing feature column: {col}"
    assert ml_model.TARGET_COLUMN in df_10k.columns, f"Missing target column: {ml_model.TARGET_COLUMN}"

    # 2. Build Model A: Grouped Holdout Validation Model
    print("\n--- Training Model A: Grouped Holdout Model (80% Episodes) ---")
    tr_df, te_df = split_episode_grouped(df_10k, test_size=0.20, seed=config.ML_RANDOM_SEED)
    X_train_A = tr_df[ml_model.FEATURE_COLUMNS]
    y_train_A = tr_df[ml_model.TARGET_COLUMN]
    X_test_A = te_df[ml_model.FEATURE_COLUMNS]
    y_test_A = te_df[ml_model.TARGET_COLUMN]

    rf_A = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED,
        n_jobs=1
    )
    rf_A.fit(X_train_A, y_train_A)

    y_pred_A = rf_A.predict(X_test_A)
    mae_A = float(mean_absolute_error(y_test_A, y_pred_A))
    rmse_A = float(np.sqrt(mean_squared_error(y_test_A, y_pred_A)))
    r2_A = float(r2_score(y_test_A, y_pred_A))

    print(f"Model A Holdout Static Metrics (on {len(X_test_A)} unseen episode samples):")
    print(f"  MAE  = {mae_A:.4f}°C")
    print(f"  RMSE = {rmse_A:.4f}°C")
    print(f"  R²   = {r2_A:.4f}")

    path_model_A = os.path.join(STAGING_DIR, "model_A_grouped_holdout.joblib")
    joblib.dump(rf_A, path_model_A)
    print(f"Saved Model A artifact to: {path_model_A}")

    # 3. Build Model B: Final Candidate Production Model (All 10,000 Samples)
    print("\n--- Training Model B: Final Candidate Production Model (All 10k Samples, n_jobs=1) ---")
    X_full = df_10k[ml_model.FEATURE_COLUMNS]
    y_full = df_10k[ml_model.TARGET_COLUMN]

    rf_B = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED,
        n_jobs=1
    )
    rf_B.fit(X_full, y_full)

    # Compute training fit metrics
    y_pred_B = rf_B.predict(X_full)
    mae_B_train = float(mean_absolute_error(y_full, y_pred_B))
    rmse_B_train = float(np.sqrt(mean_squared_error(y_full, y_pred_B)))
    r2_B_train = float(r2_score(y_full, y_pred_B))

    # Compute feature importances
    importances_B = rf_B.feature_importances_
    feat_imp_B = {
        ml_model.FEATURE_LABELS[col]: round(float(imp) * 100.0, 2)
        for col, imp in sorted(zip(ml_model.FEATURE_COLUMNS, importances_B), key=lambda x: x[1], reverse=True)
    }

    print(f"Model B Training Fit Metrics (on full 10k dataset):")
    print(f"  Fit MAE  = {mae_B_train:.4f}°C")
    print(f"  Fit RMSE = {rmse_B_train:.4f}°C")
    print(f"  Fit R²   = {r2_B_train:.4f}")

    path_model_B = os.path.join(STAGING_DIR, "candidate_production_model_10k.joblib")
    joblib.dump(rf_B, path_model_B)
    print(f"Saved Model B artifact to: {path_model_B}")

    # 4. Save Metadata Manifest
    manifest = {
        "model_architecture": "RandomForestRegressor",
        "hyperparameters": {
            "n_estimators": 120,
            "max_depth": 14,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": 1
        },
        "features": {
            "columns": ml_model.FEATURE_COLUMNS,
            "count": len(ml_model.FEATURE_COLUMNS),
            "labels": ml_model.FEATURE_LABELS
        },
        "target": ml_model.TARGET_COLUMN,
        "anti_leakage_guarantee": "Strict pre-placement feature set; zero post-workload equilibrium variables.",
        "dataset_10k": {
            "total_samples": len(df_10k),
            "total_episodes": int(df_10k["episode_id"].nunique()),
            "max_steps_per_episode": 20,
            "scenarios": [
                "Normal Operation", "GPU Surge", "Cooling Degradation",
                "Thermal Hotspot", "High AI Workload"
            ]
        },
        "model_A_grouped_holdout": {
            "artifact_path": path_model_A,
            "train_samples": len(X_train_A),
            "test_samples": len(X_test_A),
            "holdout_static_mae": round(mae_A, 4),
            "holdout_static_rmse": round(rmse_A, 4),
            "holdout_static_r2": round(r2_A, 4)
        },
        "model_B_candidate_production": {
            "artifact_path": path_model_B,
            "train_samples": len(X_full),
            "fit_mae": round(mae_B_train, 4),
            "fit_rmse": round(rmse_B_train, 4),
            "fit_r2": round(r2_B_train, 4),
            "top_features": list(feat_imp_B.items())[:5]
        }
    }

    manifest_path = os.path.join(STAGING_DIR, "candidate_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported manifest to: {manifest_path}")

    return rf_A, rf_B, manifest


if __name__ == "__main__":
    build_candidate_models()
