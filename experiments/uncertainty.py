"""
HeatWise Experiment F: Tree-Variance Uncertainty Estimation
Zero-dependency ensemble disagreement estimator using RandomForestRegressor.estimators_.
Evaluates risk-averse thermal placement bounds: T_conservative = mu + k * sigma.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import ml_model
from experiments.dataset_utils import generate_sequential_dataset
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


class UncertaintyRandomForestPredictor:
    """
    Wrapper around RandomForestRegressor exposing per-tree ensemble variance.
    Pure NumPy implementation: zero additional dependencies.
    """
    def __init__(self, model: RandomForestRegressor):
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Computes ensemble mean and sample standard deviation across all trees.
        Returns: (mean_prediction, sample_std_dev)
        """
        vals = X.values if hasattr(X, "values") else X
        # shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(vals) for tree in self.model.estimators_])
        mean_val = float(np.mean(tree_preds, axis=0)[0])
        std_val = float(np.std(tree_preds, axis=0, ddof=1)[0])
        return mean_val, std_val


def benchmark_uncertainty_latency(wrapper: UncertaintyRandomForestPredictor, X_sample: pd.DataFrame, n_trials: int = 500) -> Dict[str, float]:
    """Measures actual CPU latency of standard predict vs. tree-by-tree uncertainty extraction."""
    # Standard predict
    t0 = time.time()
    for _ in range(n_trials):
        _ = wrapper.predict(X_sample)
    std_lat_ms = (time.time() - t0) / n_trials * 1000.0

    # Tree-by-tree predict
    t1 = time.time()
    for _ in range(n_trials):
        _, _ = wrapper.predict_with_uncertainty(X_sample)
    unc_lat_ms = (time.time() - t1) / n_trials * 1000.0

    return {
        "standard_predict_ms": round(std_lat_ms, 3),
        "uncertainty_predict_ms": round(unc_lat_ms, 3),
        "overhead_factor": round(unc_lat_ms / max(0.001, std_lat_ms), 2)
    }


def run_experiment_uncertainty(dataset_path: str = None) -> List[Dict[str, Any]]:
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

    print("\n--- [EXP F] Training Random Forest for Uncertainty Profiling ---")
    base_rf = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    base_rf.fit(X_train, y_train)

    wrapper = UncertaintyRandomForestPredictor(base_rf)

    # Benchmark actual latency
    print("Benchmarking inference latency on 12-rack batch...")
    sample_12 = X_test.iloc[0:1]
    latency_profile = benchmark_uncertainty_latency(wrapper, sample_12, n_trials=500)
    print(f"Latency Profile: {latency_profile}")

    k_values = [0.0, 1.0, 1.28, 1.64, 2.0]
    results = []

    for k in k_values:
        print(f"\n--- Testing Risk-Aversion k = {k} ---")
        eval_res = run_dynamic_closed_loop_eval(
            wrapper,
            n_decisions_per_scenario=50,
            seed=42,
            conservative_k=k
        )

        entry = {
            "k_multiplier": k,
            "interpretation": "Standard Point Prediction" if k == 0.0 else f"Mean + {k}*Sigma Bound",
            "unsafe_placements": eval_res["unsafe_placements_count"],
            "threshold_edge_decisions": eval_res["threshold_edge_decisions"],
            "rejected_by_conservative_bound": eval_res["rejected_by_conservative_bound"],
            "dynamic_mae": eval_res["dynamic_mae"],
            "optimality_gap": eval_res["optimality_gap"],
            "top1_agreement_pct": eval_res["top1_agreement_pct"],
            "top3_overlap_pct": eval_res["top3_overlap_pct"],
            "materially_worse_rate_pct": eval_res["materially_worse_rate_pct"],
            "standard_predict_ms": latency_profile["standard_predict_ms"],
            "uncertainty_predict_ms": latency_profile["uncertainty_predict_ms"]
        }
        results.append(entry)
        print(f"k={k}: Unsafe={eval_res['unsafe_placements_count']}, RejectedByBound={eval_res['rejected_by_conservative_bound']}, Top-1={eval_res['top1_agreement_pct']}%, OptGap={eval_res['optimality_gap']}°C")

    # Save summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "uncertainty.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "uncertainty.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment_uncertainty()
