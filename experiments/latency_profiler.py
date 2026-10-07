"""
HeatWise Phase 11: Comprehensive System Latency Profiler
Measures real CPU execution latencies across single predictions, tree variance extraction,
12-rack scoring, end-to-end placement, and dataset generation on the local machine.
"""

import sys
import os
import time
import platform
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import sklearn
import scipy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import simulation
import heatwise
import ml_model
from experiments.uncertainty import UncertaintyRandomForestPredictor

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def profile_latencies(n_iterations: int = 300) -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. System Metadata
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "total_racks": config.TOTAL_RACKS,
        "n_trees": 120
    }

    print("\n--- System & Hardware Environment ---")
    for k, v in system_info.items():
        print(f"  {k}: {v}")

    # Prepare simulation, engine, and test workload
    sim = simulation.DataCenterSimulation(scenario="Thermal Hotspot")
    predictor = ml_model.get_trained_predictor()
    engine = heatwise.HeatWiseDecisionEngine(predictor=predictor)
    unc_wrapper = UncertaintyRandomForestPredictor(predictor.model)

    wl = heatwise.Workload("AI Training", 3, 50.0, 50.0)
    rack = sim.racks[1]
    single_feat_df = ml_model.build_prediction_features(rack, wl)
    batch_feat_df = pd.concat([ml_model.build_prediction_features(sim.racks[r], wl) for r in range(1, 13)], ignore_index=True)

    benchmarks = {}

    # Benchmark 1: Standard RF Single-Rack Predict
    latencies_rf_single = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = predictor.model.predict(single_feat_df)
        latencies_rf_single.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["standard_rf_single_rack_ms"] = latencies_rf_single

    # Benchmark 2: Standard RF 12-Rack Batch Predict
    latencies_rf_12 = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = predictor.model.predict(batch_feat_df)
        latencies_rf_12.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["standard_rf_12_rack_batch_ms"] = latencies_rf_12

    # Benchmark 3: Tree-by-Tree Uncertainty Extraction (Single Rack)
    latencies_unc_single = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _, _ = unc_wrapper.predict_with_uncertainty(single_feat_df)
        latencies_unc_single.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["tree_variance_single_rack_ms"] = latencies_unc_single

    # Benchmark 4: Tree-by-Tree Uncertainty Extraction (12 Racks)
    latencies_unc_12 = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        vals = batch_feat_df.values
        _ = np.array([tree.predict(vals) for tree in unc_wrapper.model.estimators_])
        latencies_unc_12.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["tree_variance_12_racks_ms"] = latencies_unc_12

    # Benchmark 5: Full 12-Rack Candidate Scoring (Heuristic)
    latencies_scoring = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        for r_id in range(1, 13):
            headroom = max(0.0, config.TEMP_CRITICAL - 65.0)
            sub_temp = heatwise.score_temperature_safety(65.0)
            sub_head = heatwise.score_thermal_headroom(headroom)
            sub_gpu = heatwise.score_gpu_availability(50.0)
            sub_cool = heatwise.score_cooling_capacity(90.0)
            sub_nbr = heatwise.score_neighbor_stability(60.0)
            _ = 0.35 * sub_temp + 0.25 * sub_head + 0.20 * sub_gpu + 0.10 * sub_cool + 0.10 * sub_nbr
        latencies_scoring.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["candidate_scoring_12_racks_ms"] = latencies_scoring

    # Benchmark 6: Full Placement Decision Loop (End-to-End)
    latencies_e2e = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = engine.evaluate_placement(sim, wl)
        latencies_e2e.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["full_placement_decision_e2e_ms"] = latencies_e2e

    # Benchmark 7: Dataset Generation (10 transition steps = 120 samples)
    latencies_gen_step = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = simulation.generate_synthetic_training_data(n_samples=120)
        latencies_gen_step.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["dataset_generation_120_samples_ms"] = latencies_gen_step

    summary_records = []
    print("\n--- Benchmark Latency Summary (ms) ---")
    for name, vals in benchmarks.items():
        arr = np.array(vals)
        rec = {
            "operation": name,
            "mean_ms": round(float(np.mean(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "std_ms": round(float(np.std(arr)), 3),
            "iterations": len(arr)
        }
        summary_records.append(rec)
        print(f"  {name:35s}: mean={rec['mean_ms']:7.3f}ms  median={rec['median_ms']:7.3f}ms  p95={rec['p95_ms']:7.3f}ms  p99={rec['p99_ms']:7.3f}ms")

    # Save outputs
    df_lat = pd.DataFrame(summary_records)
    df_lat.to_csv(os.path.join(RESULTS_DIR, "latency.csv"), index=False)

    full_output = {
        "system_info": system_info,
        "latency_summary": summary_records
    }
    with open(os.path.join(RESULTS_DIR, "latency.json"), "w") as f:
        json.dump(full_output, f, indent=2)

    return full_output


if __name__ == "__main__":
    profile_latencies()
