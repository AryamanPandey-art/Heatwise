"""
HeatWise Dynamic Evaluation Utilities
Standardized closed-loop evaluation harness comparing candidate placements
against the authoritative physical simulation oracle across 250 decisions.
"""

import sys
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import simulation
import heatwise
import ml_model


def run_dynamic_closed_loop_eval(
    predictor_or_model,
    n_decisions_per_scenario: int = 50,
    seed: int = 42,
    conservative_k: float = 0.0,
    feature_columns: List[str] = None,
    custom_scoring_fn = None,
    target_scenarios: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluates placement decisions across specified operational scenarios.
    Compares the surrogate model's recommendation against the physical simulation oracle.
    """
    if feature_columns is None:
        feature_columns = ml_model.FEATURE_COLUMNS

    if target_scenarios is None:
        scenarios = ["Normal Operation", "GPU Surge", "Cooling Degradation", "Thermal Hotspot", "High AI Workload"]
    else:
        scenarios = target_scenarios

    total_decisions = len(scenarios) * n_decisions_per_scenario

    top1s = 0
    top3s = 0
    spears = []
    mae_list = []
    ml_actual_temps = []
    oracle_actual_temps = []
    materially_worse_count = 0
    unsafe_placements_count = 0  # >= 80°C
    hw_infeasible_count = 0
    scenario_worse = {s: 0 for s in scenarios}
    threshold_edge_decisions = 0  # True temp between 73.5°C and 76.5°C
    rejected_by_conservative_bound = 0

    rng = np.random.RandomState(seed)

    for sc in scenarios:
        for _ in range(n_decisions_per_scenario):
            sim = simulation.DataCenterSimulation(scenario=sc)

            # Background state evolution: inject 10 random workloads
            for _ in range(10):
                wl_p = float(rng.uniform(500, 2000))
                win = int(rng.randint(1, 13))
                sim.racks[win]["total_power_w"] += wl_p
                for _ in range(4):
                    te = {r: sim.racks[r]["temperature"] for r in range(1, 13)}
                    for r in range(1, 13):
                        nbr = [te[nid] for nid in sim.racks[r]["neighbor_ids"]]
                        _, _, teq = simulation.compute_equilibrium_temperature(
                            sim.racks[r]["total_power_w"],
                            sim.racks[r]["cooling_capacity_pct"],
                            np.mean(nbr),
                            sim.racks[r]["recirculation_factor"]
                        )
                        sim.racks[r]["temperature"] = teq
                        sim.racks[r]["neighbor_avg_temp"] = np.mean(nbr)

            # Evaluated test workload
            wl_gpu = float(rng.choice([20.0, 40.0, 60.0]))
            wl = heatwise.Workload(
                workload_type="AI Training",
                intensity=3,
                gpu_demand_pct=wl_gpu,
                memory_demand_pct=30.0
            )

            candidate_evals = []
            c_oracle = []

            for r_id in range(1, 13):
                rack = sim.racks[r_id]

                # Check hardware feasibility
                has_gpu = rack["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0)
                has_mem = rack["mem_available_pct"] >= (wl.memory_demand_pct / 4.0)

                # Build features
                feat_df = ml_model.build_prediction_features(rack, wl)
                feat_input = feat_df[feature_columns]

                # Predict temperature
                pred_std = 0.0
                if hasattr(predictor_or_model, "predict_with_uncertainty"):
                    pred_temp, pred_std = predictor_or_model.predict_with_uncertainty(feat_input)
                elif hasattr(predictor_or_model, "predict"):
                    pred_temp = float(predictor_or_model.predict(feat_input)[0])
                elif hasattr(predictor_or_model, "predict_rack_placement"):
                    pred_temp, _ = predictor_or_model.predict_rack_placement(rack, wl)
                else:
                    raise ValueError("Unsupported predictor/model object.")

                # Conservative upper bound if k > 0
                effective_pred_temp = pred_temp + conservative_k * pred_std
                if conservative_k > 0 and pred_temp < config.TEMP_CRITICAL and effective_pred_temp >= config.TEMP_CRITICAL:
                    rejected_by_conservative_bound += 1

                # Physical simulation ground truth for candidate
                sim_res = sim.simulate_placement(r_id, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
                actual_temp = sim_res["new_temp"]

                if 73.5 <= actual_temp <= 76.5:
                    threshold_edge_decisions += 1

                # Scoring
                if custom_scoring_fn:
                    score, is_rejected = custom_scoring_fn(rack, effective_pred_temp, has_gpu, has_mem)
                else:
                    headroom = max(0.0, config.TEMP_CRITICAL - effective_pred_temp)
                    sub_temp = heatwise.score_temperature_safety(effective_pred_temp)
                    sub_head = heatwise.score_thermal_headroom(headroom)
                    sub_gpu = heatwise.score_gpu_availability(rack["gpu_available_pct"])
                    sub_cool = heatwise.score_cooling_capacity(rack["cooling_capacity_pct"])
                    sub_nbr = heatwise.score_neighbor_stability(rack["neighbor_avg_temp"])

                    raw_score = (
                        0.35 * sub_temp +
                        0.25 * sub_head +
                        0.20 * sub_gpu +
                        0.10 * sub_cool +
                        0.10 * sub_nbr
                    )

                    penalty = 0.0
                    is_hotspot = effective_pred_temp >= config.TEMP_CRITICAL
                    if is_hotspot:
                        penalty += 45.0 + (effective_pred_temp - config.TEMP_CRITICAL) * 5.0
                    if not has_gpu:
                        penalty += 50.0
                    if not has_mem:
                        penalty += 50.0

                    score = float(np.clip(raw_score - penalty, 0.0, 100.0))
                    is_rejected = (not has_gpu) or (not has_mem) or is_hotspot

                candidate_evals.append({
                    "rack_id": r_id,
                    "pred_temp": pred_temp,
                    "effective_pred_temp": effective_pred_temp,
                    "actual_temp": actual_temp,
                    "score": score,
                    "is_rejected": is_rejected,
                    "has_hardware": has_gpu and has_mem
                })

                if has_gpu and has_mem:
                    c_oracle.append((r_id, actual_temp))

            # Sort candidate selections
            feasible_ml = [c for c in candidate_evals if not c["is_rejected"]]
            if feasible_ml:
                best_ml = max(feasible_ml, key=lambda x: x["score"])
            else:
                hw_feasible = [c for c in candidate_evals if c["has_hardware"]]
                best_ml = max(hw_feasible, key=lambda x: x["score"]) if hw_feasible else candidate_evals[0]

            # Oracle selection: lowest actual temperature among hardware feasible racks
            if c_oracle:
                c_oracle.sort(key=lambda x: x[1])
                best_oracle_id, best_oracle_actual_temp = c_oracle[0]
            else:
                best_oracle_id = best_ml["rack_id"]
                best_oracle_actual_temp = best_ml["actual_temp"]

            # Rank ML candidates by score descending
            ranked_ml = sorted(candidate_evals, key=lambda x: x["score"], reverse=True)
            ranked_oracle = sorted(candidate_evals, key=lambda x: x["actual_temp"])

            ml_ranks = [c["rack_id"] for c in ranked_ml]
            oracle_ranks = [c["rack_id"] for c in ranked_oracle]

            # Compute step metrics
            if best_ml["rack_id"] == best_oracle_id:
                top1s += 1
            if best_oracle_id in ml_ranks[:3]:
                top3s += 1

            rho, _ = spearmanr(ml_ranks, oracle_ranks)
            if not np.isnan(rho):
                spears.append(rho)

            mae_list.append(np.mean([abs(c["pred_temp"] - c["actual_temp"]) for c in candidate_evals]))
            ml_actual_temps.append(best_ml["actual_temp"])
            oracle_actual_temps.append(best_oracle_actual_temp)

            diff = best_ml["actual_temp"] - best_oracle_actual_temp
            if diff > 2.0:
                materially_worse_count += 1
                scenario_worse[sc] += 1

            if best_ml["actual_temp"] >= config.TEMP_MAX_SAFE:
                unsafe_placements_count += 1

            if not best_ml["has_hardware"]:
                hw_infeasible_count += 1

    opt_gap = float(np.mean(ml_actual_temps) - np.mean(oracle_actual_temps))

    return {
        "dynamic_mae": round(float(np.mean(mae_list)), 3),
        "top1_agreement_pct": round(float(top1s / total_decisions * 100.0), 2),
        "top3_overlap_pct": round(float(top3s / total_decisions * 100.0), 2),
        "spearman_rho": round(float(np.mean(spears)), 3),
        "ml_mean_actual_temp": round(float(np.mean(ml_actual_temps)), 2),
        "oracle_mean_actual_temp": round(float(np.mean(oracle_actual_temps)), 2),
        "optimality_gap": round(opt_gap, 3),
        "materially_worse_rate_pct": round(float(materially_worse_count / total_decisions * 100.0), 2),
        "unsafe_placements_count": unsafe_placements_count,
        "hw_infeasible_count": hw_infeasible_count,
        "total_decisions": total_decisions,
        "threshold_edge_decisions": threshold_edge_decisions,
        "rejected_by_conservative_bound": rejected_by_conservative_bound,
        "scenario_worse_breakdown": scenario_worse
    }
