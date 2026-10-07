"""
HeatWise Robustness and Production Readiness Suite
Executes:
1. Experiment 1: Multi-Seed Dynamic Replication (Seeds: 101, 202, 303, 404, 505)
   comparing 10k episode-grouped RF vs 20k episode-grouped RF across 250 decisions.
   Reports per-seed metrics and Mean ± SD for both models.
2. Experiment 4: Latency Optimization Safety Check (n_jobs=-1 vs n_jobs=1)
   Verifying exact numerical prediction equality and placement decision identity,
   alongside micro-benchmark latency profiling (mean, median, P95, P99).
Outputs:
- experiments/results/final_robustness_check.csv
- experiments/results/final_robustness_check.json
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import simulation
import ml_model
import heatwise
from experiments.dataset_utils import split_episode_grouped
from experiments.eval_utils import run_dynamic_closed_loop_eval

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
EVAL_SEEDS = [101, 202, 303, 404, 505]


def run_multi_seed_experiment(df_10k: pd.DataFrame, df_20k: pd.DataFrame):
    print("================================================================================")
    print("EXPERIMENT 1: MULTI-SEED DYNAMIC REPLICATION (Seeds: 101, 202, 303, 404, 505)")
    print("================================================================================")

    # 1. Train 10k Episode-Grouped Model
    print("Splitting and training 10k Episode-Grouped RF (120 trees)...")
    tr_10k, te_10k = split_episode_grouped(df_10k, test_size=0.20, seed=config.ML_RANDOM_SEED)
    rf_10k = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=1
    )
    rf_10k.fit(tr_10k[ml_model.FEATURE_COLUMNS], tr_10k[ml_model.TARGET_COLUMN])

    # 2. Train 20k Episode-Grouped Model
    print("Splitting and training 20k Episode-Grouped RF (120 trees)...")
    tr_20k, te_20k = split_episode_grouped(df_20k, test_size=0.20, seed=config.ML_RANDOM_SEED)
    rf_20k = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=1
    )
    rf_20k.fit(tr_20k[ml_model.FEATURE_COLUMNS], tr_20k[ml_model.TARGET_COLUMN])

    eval_records = []

    models = [
        ("10k_episode_grouped", rf_10k, te_10k),
        ("20k_episode_grouped", rf_20k, te_20k)
    ]

    for model_name, model_inst, te_df in models:
        print(f"\n--- Evaluating Model: {model_name} ---")
        for s in EVAL_SEEDS:
            t0 = time.time()
            res = run_dynamic_closed_loop_eval(
                model_inst,
                n_decisions_per_scenario=50,
                seed=s,
                conservative_k=0.0
            )
            elapsed = round(time.time() - t0, 2)
            rec = {
                "model": model_name,
                "eval_seed": s,
                "decisions": res["total_decisions"],
                "dynamic_mae": res["dynamic_mae"],
                "top1_agreement_pct": res["top1_agreement_pct"],
                "top3_overlap_pct": res["top3_overlap_pct"],
                "spearman_rho": res["spearman_rho"],
                "optimality_gap": res["optimality_gap"],
                "unsafe_placements": res["unsafe_placements_count"],
                "hw_infeasible": res["hw_infeasible_count"],
                "materially_worse_pct": res["materially_worse_rate_pct"],
                "eval_time_s": elapsed
            }
            eval_records.append(rec)
            print(f"  Seed {s} [{elapsed}s]: MAE={rec['dynamic_mae']}°C, Top1={rec['top1_agreement_pct']}%, Top3={rec['top3_overlap_pct']}%, rho={rec['spearman_rho']}, Gap={rec['optimality_gap']}°C, Unsafe={rec['unsafe_placements']}, Infeasible={rec['hw_infeasible']}")

    # Compute Summary Statistics (Mean ± SD)
    summary_stats = {}
    for model_name in ["10k_episode_grouped", "20k_episode_grouped"]:
        m_recs = [r for r in eval_records if r["model"] == model_name]
        summary_stats[model_name] = {
            "dynamic_mae_mean": round(float(np.mean([r["dynamic_mae"] for r in m_recs])), 3),
            "dynamic_mae_sd": round(float(np.std([r["dynamic_mae"] for r in m_recs], ddof=1)), 3),
            "top1_agreement_mean": round(float(np.mean([r["top1_agreement_pct"] for r in m_recs])), 2),
            "top1_agreement_sd": round(float(np.std([r["top1_agreement_pct"] for r in m_recs], ddof=1)), 2),
            "top3_overlap_mean": round(float(np.mean([r["top3_overlap_pct"] for r in m_recs])), 2),
            "top3_overlap_sd": round(float(np.std([r["top3_overlap_pct"] for r in m_recs], ddof=1)), 2),
            "spearman_rho_mean": round(float(np.mean([r["spearman_rho"] for r in m_recs])), 3),
            "spearman_rho_sd": round(float(np.std([r["spearman_rho"] for r in m_recs], ddof=1)), 3),
            "optimality_gap_mean": round(float(np.mean([r["optimality_gap"] for r in m_recs])), 3),
            "optimality_gap_sd": round(float(np.std([r["optimality_gap"] for r in m_recs], ddof=1)), 3),
            "unsafe_placements_total": sum(r["unsafe_placements"] for r in m_recs),
            "hw_infeasible_total": sum(r["hw_infeasible"] for r in m_recs),
            "materially_worse_mean": round(float(np.mean([r["materially_worse_pct"] for r in m_recs])), 2),
            "materially_worse_sd": round(float(np.std([r["materially_worse_pct"] for r in m_recs], ddof=1)), 2)
        }

    return eval_records, summary_stats, rf_10k, tr_10k, te_10k


def run_latency_safety_check(rf_ser: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    print("\n================================================================================")
    print("EXPERIMENT 4: LATENCY OPTIMIZATION & SAFETY CHECK (n_jobs=-1 vs n_jobs=1)")
    print("================================================================================")

    # Train parallel model with identical seed & params
    print("Training parallel model (n_jobs=-1)...")
    rf_par = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=-1
    )
    # Train on first 8000 samples for swift identical structure
    X_sample = X_test[ml_model.FEATURE_COLUMNS]
    rf_par.fit(X_sample, y_test.iloc[:len(X_sample)])
    rf_ser_cmp = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_split=4, min_samples_leaf=2,
        random_state=config.ML_RANDOM_SEED, n_jobs=1
    )
    rf_ser_cmp.fit(X_sample, y_test.iloc[:len(X_sample)])

    # 1. Numerical Equivalence of Model Output
    print("Testing numerical output identity on test set...")
    pred_ser = rf_ser_cmp.predict(X_sample)
    pred_par = rf_par.predict(X_sample)

    max_abs_diff = float(np.max(np.abs(pred_ser - pred_par)))
    mean_abs_diff = float(np.mean(np.abs(pred_ser - pred_par)))
    is_exact_match = bool(np.allclose(pred_ser, pred_par, atol=1e-9))
    print(f"  Max absolute prediction difference: {max_abs_diff:.2e}°C")
    print(f"  Mean absolute prediction difference: {mean_abs_diff:.2e}°C")
    print(f"  Numerically identical: {is_exact_match}")

    # 2. Decision Engine Identity across Dynamic Closed-Loop Decisions
    print("Evaluating decision equivalence on 250 closed-loop placement steps...")
    eng_ser = heatwise.HeatWiseDecisionEngine(predictor=ml_model.ThermalMLPredictor())
    eng_ser.predictor.model = rf_ser_cmp
    eng_ser.predictor.is_trained = True

    eng_par = heatwise.HeatWiseDecisionEngine(predictor=ml_model.ThermalMLPredictor())
    eng_par.predictor.model = rf_par
    eng_par.predictor.is_trained = True

    scenarios = ["Normal Operation", "GPU Surge", "Cooling Degradation", "Thermal Hotspot", "High AI Workload"]
    decision_matches = 0
    total_steps = 0
    score_diffs = []

    for sc in scenarios:
        sim_ser = simulation.DataCenterSimulation(scenario=sc)
        sim_par = simulation.DataCenterSimulation(scenario=sc)
        for step in range(50):
            total_steps += 1
            wl_gpu = float(np.random.choice([20.0, 40.0, 60.0]))
            wl = heatwise.Workload(workload_type="AI Training", intensity=3, gpu_demand_pct=wl_gpu, memory_demand_pct=30.0)

            # Evaluate with serial engine
            eval_ser = eng_ser.evaluate_placement(sim_ser, wl)
            # Evaluate with parallel engine
            eval_par = eng_par.evaluate_placement(sim_par, wl)

            r_ser = eval_ser["recommended_rack"]["rack_id"]
            r_par = eval_par["recommended_rack"]["rack_id"]
            sc_ser = eval_ser["recommended_rack"]["heatwise_score"]
            sc_par = eval_par["recommended_rack"]["heatwise_score"]

            score_diffs.append(abs(sc_ser - sc_par))
            if r_ser == r_par:
                decision_matches += 1

            # Advance state equally
            sim_ser.advance_time(1)
            sim_par.advance_time(1)

    decision_identity_pct = round(decision_matches / total_steps * 100.0, 2)
    max_score_diff = float(np.max(score_diffs))
    print(f"  Selected rack identity: {decision_matches}/{total_steps} ({decision_identity_pct}%)")
    print(f"  Max score difference: {max_score_diff}")

    # 3. Latency Micro-benchmarking (200 iterations each)
    print("Micro-benchmarking latency breakdown across 200 iterations...")
    n_iters = 200
    sample_1 = X_sample.iloc[0:1]
    sample_12 = X_sample.iloc[0:12]

    # Helper function for percentiles
    def get_stats(arr):
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3)
        }

    # A. Single sample predict
    lat_single_ser = []
    lat_single_par = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = rf_ser_cmp.predict(sample_1)
        lat_single_ser.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = rf_par.predict(sample_1)
        lat_single_par.append((time.perf_counter() - t0) * 1000.0)

    # B. 12-rack batch predict
    lat_12_ser = []
    lat_12_par = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = rf_ser_cmp.predict(sample_12)
        lat_12_ser.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = rf_par.predict(sample_12)
        lat_12_par.append((time.perf_counter() - t0) * 1000.0)

    # C. Feature construction (12 racks)
    sim_bench = simulation.DataCenterSimulation(scenario="Thermal Hotspot")
    wl_bench = heatwise.Workload("AI Training", 3, 50.0, 50.0)
    lat_feat = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = [ml_model.build_prediction_features(sim_bench.racks[r], wl_bench) for r in range(1, 13)]
        lat_feat.append((time.perf_counter() - t0) * 1000.0)

    # D. Candidate scoring (12 racks)
    lat_score = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        for r_id in range(1, 13):
            sub_temp = heatwise.score_temperature_safety(65.0)
            sub_head = heatwise.score_thermal_headroom(10.0)
            sub_gpu = heatwise.score_gpu_availability(50.0)
            sub_cool = heatwise.score_cooling_capacity(90.0)
            sub_nbr = heatwise.score_neighbor_stability(60.0)
            _ = 0.35 * sub_temp + 0.25 * sub_head + 0.20 * sub_gpu + 0.10 * sub_cool + 0.10 * sub_nbr
        lat_score.append((time.perf_counter() - t0) * 1000.0)

    # E. Full E2E placement loop (12 candidate evaluations + ranking)
    lat_e2e_ser = []
    lat_e2e_par = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = eng_ser.evaluate_placement(sim_bench, wl_bench)
        lat_e2e_ser.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = eng_par.evaluate_placement(sim_bench, wl_bench)
        lat_e2e_par.append((time.perf_counter() - t0) * 1000.0)

    latency_results = {
        "equivalence": {
            "max_abs_prediction_diff_c": max_abs_diff,
            "mean_abs_prediction_diff_c": mean_abs_diff,
            "numerically_identical": is_exact_match,
            "decision_identity_pct": decision_identity_pct,
            "max_score_diff": max_score_diff
        },
        "single_predict_serial_ms": get_stats(lat_single_ser),
        "single_predict_parallel_ms": get_stats(lat_single_par),
        "batch_12_predict_serial_ms": get_stats(lat_12_ser),
        "batch_12_predict_parallel_ms": get_stats(lat_12_par),
        "feature_construction_12_racks_ms": get_stats(lat_feat),
        "candidate_scoring_12_racks_ms": get_stats(lat_score),
        "full_e2e_placement_serial_jobs_1_ms": get_stats(lat_e2e_ser),
        "full_e2e_placement_parallel_jobs_neg1_ms": get_stats(lat_e2e_par)
    }

    print("\n--- Latency Breakdown Summary ---")
    print(f"  Feature Construction (12 racks): Mean={latency_results['feature_construction_12_racks_ms']['mean_ms']} ms, P95={latency_results['feature_construction_12_racks_ms']['p95_ms']} ms")
    print(f"  Candidate Scoring (12 racks):    Mean={latency_results['candidate_scoring_12_racks_ms']['mean_ms']} ms, P95={latency_results['candidate_scoring_12_racks_ms']['p95_ms']} ms")
    print(f"  12-Rack Predict (n_jobs=1):       Mean={latency_results['batch_12_predict_serial_ms']['mean_ms']} ms, P95={latency_results['batch_12_predict_serial_ms']['p95_ms']} ms")
    print(f"  12-Rack Predict (n_jobs=-1):      Mean={latency_results['batch_12_predict_parallel_ms']['mean_ms']} ms, P95={latency_results['batch_12_predict_parallel_ms']['p95_ms']} ms")
    print(f"  E2E Placement Loop (n_jobs=1):    Mean={latency_results['full_e2e_placement_serial_jobs_1_ms']['mean_ms']} ms, P95={latency_results['full_e2e_placement_serial_jobs_1_ms']['p95_ms']} ms")
    print(f"  E2E Placement Loop (n_jobs=-1):   Mean={latency_results['full_e2e_placement_parallel_jobs_neg1_ms']['mean_ms']} ms, P95={latency_results['full_e2e_placement_parallel_jobs_neg1_ms']['p95_ms']} ms")

    return latency_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_10k_path = os.path.join(RESULTS_DIR, "dataset_A2_10000.csv")
    df_20k_path = os.path.join(RESULTS_DIR, "dataset_A3_20000.csv")

    print(f"Loading datasets from {RESULTS_DIR}...")
    df_10k = pd.read_csv(df_10k_path)
    df_20k = pd.read_csv(df_20k_path)

    # 1. Multi-Seed Replication
    eval_records, summary_stats, rf_10k, tr_10k, te_10k = run_multi_seed_experiment(df_10k, df_20k)

    # 2. Latency Optimization Safety Check
    latency_results = run_latency_safety_check(rf_10k, te_10k, tr_10k[ml_model.TARGET_COLUMN])

    # 3. Export CSV and JSON
    df_results = pd.DataFrame(eval_records)
    csv_path = os.path.join(RESULTS_DIR, "final_robustness_check.csv")
    json_path = os.path.join(RESULTS_DIR, "final_robustness_check.json")

    df_results.to_csv(csv_path, index=False)

    full_payload = {
        "multi_seed_records": eval_records,
        "summary_statistics": summary_stats,
        "latency_and_safety": latency_results
    }

    with open(json_path, "w") as f:
        json.dump(full_payload, f, indent=2)

    print(f"\n================================================================================")
    print(f"SUCCESS: Exported results to:\n  {csv_path}\n  {json_path}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
