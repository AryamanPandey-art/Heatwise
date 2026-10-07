"""
HeatWise Extended Benchmarks & Stress Testing Suite
Evaluates multi-seed robustness (Exp H), bursty arrivals (Exp I),
high concurrency (Exp J), and sustained high utilization (Exp K).
Compares Round Robin, Least Loaded, and HeatWise across all regimes.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import simulation
import heatwise
import ml_model

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def run_single_simulation_stream(
    workload_stream: List[heatwise.Workload],
    base_scenario: str = "Thermal Hotspot",
    engine: heatwise.HeatWiseDecisionEngine = None
) -> Dict[str, Dict[str, Any]]:
    """
    Executes a stream of workloads synchronously across Round Robin, Least Loaded, and HeatWise.
    Returns comprehensive metrics for each scheduler.
    """
    if engine is None:
        predictor = ml_model.get_trained_predictor()
        engine = heatwise.HeatWiseDecisionEngine(predictor=predictor)

    sim_rr = simulation.DataCenterSimulation(scenario=base_scenario)
    sim_ll = simulation.DataCenterSimulation(scenario=base_scenario)
    sim_hw = simulation.DataCenterSimulation(scenario=base_scenario)

    metrics = {
        "Round Robin": {"placement_failures": 0, "fallback_invocations": 0},
        "Least Loaded": {"placement_failures": 0, "fallback_invocations": 0},
        "HeatWise": {"placement_failures": 0, "fallback_invocations": 0}
    }

    rr_idx = 1
    num_workloads = len(workload_stream)

    for wl in workload_stream:
        # 1. Round Robin
        rr_chosen = None
        for _ in range(config.TOTAL_RACKS):
            candidate_id = rr_idx
            rr_idx = (rr_idx % config.TOTAL_RACKS) + 1
            rack = sim_rr.racks[candidate_id]
            if rack["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0):
                rr_chosen = candidate_id
                break
        if rr_chosen is None:
            rr_chosen = 1

        rr_rack = sim_rr.racks[rr_chosen]
        if rr_rack["mem_available_pct"] < (wl.memory_demand_pct / 4.0) or rr_rack["gpu_available_pct"] < (wl.gpu_demand_pct / 4.0):
            metrics["Round Robin"]["placement_failures"] += 1
        else:
            res = sim_rr.simulate_placement(rr_chosen, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_rr.racks[rr_chosen]["total_power_w"] = res["new_power_w"]
            sim_rr.racks[rr_chosen]["avg_gpu_util"] = res["new_gpu_util"]
            sim_rr.racks[rr_chosen]["avg_mem_util"] = res["new_mem_util"]
            sim_rr.racks[rr_chosen]["temperature"] = res["new_temp"]
            sim_rr.racks[rr_chosen]["gpu_available_pct"] = round(100.0 - res["new_gpu_util"], 1)
            sim_rr.racks[rr_chosen]["mem_available_pct"] = round(100.0 - res["new_mem_util"], 1)
            sim_rr.racks[rr_chosen].setdefault("active_workloads", []).append({
                "gpu_pct": wl.gpu_demand_pct, "mem_pct": wl.memory_demand_pct, "power_w": wl.power_est_w,
                "duration_remaining": max(1, wl.duration_s // 15)
            })

        # 2. Least Loaded
        ll_cands = [
            (r_id, r["gpu_available_pct"])
            for r_id, r in sim_ll.racks.items()
            if r["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0)
        ]
        if ll_cands:
            ll_chosen = max(ll_cands, key=lambda x: x[1])[0]
        else:
            ll_chosen = max(sim_ll.racks.items(), key=lambda x: x[1]["gpu_available_pct"])[0]

        ll_rack = sim_ll.racks[ll_chosen]
        if ll_rack["mem_available_pct"] < (wl.memory_demand_pct / 4.0) or ll_rack["gpu_available_pct"] < (wl.gpu_demand_pct / 4.0):
            metrics["Least Loaded"]["placement_failures"] += 1
        else:
            res = sim_ll.simulate_placement(ll_chosen, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_ll.racks[ll_chosen]["total_power_w"] = res["new_power_w"]
            sim_ll.racks[ll_chosen]["avg_gpu_util"] = res["new_gpu_util"]
            sim_ll.racks[ll_chosen]["avg_mem_util"] = res["new_mem_util"]
            sim_ll.racks[ll_chosen]["temperature"] = res["new_temp"]
            sim_ll.racks[ll_chosen]["gpu_available_pct"] = round(100.0 - res["new_gpu_util"], 1)
            sim_ll.racks[ll_chosen]["mem_available_pct"] = round(100.0 - res["new_mem_util"], 1)
            sim_ll.racks[ll_chosen].setdefault("active_workloads", []).append({
                "gpu_pct": wl.gpu_demand_pct, "mem_pct": wl.memory_demand_pct, "power_w": wl.power_est_w,
                "duration_remaining": max(1, wl.duration_s // 15)
            })

        # 3. HeatWise
        hw_eval = engine.evaluate_placement(sim_hw, wl)
        hw_rec = hw_eval["recommended_rack"]
        hw_chosen = hw_rec["rack_id"]

        if hw_rec["is_rejected"]:
            metrics["HeatWise"]["placement_failures"] += 1
            if hw_rec.get("is_hotspot_risk", False):
                metrics["HeatWise"]["fallback_invocations"] += 1
        else:
            res = sim_hw.simulate_placement(hw_chosen, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_hw.racks[hw_chosen]["total_power_w"] = res["new_power_w"]
            sim_hw.racks[hw_chosen]["avg_gpu_util"] = res["new_gpu_util"]
            sim_hw.racks[hw_chosen]["avg_mem_util"] = res["new_mem_util"]
            sim_hw.racks[hw_chosen]["temperature"] = res["new_temp"]
            sim_hw.racks[hw_chosen]["gpu_available_pct"] = round(100.0 - res["new_gpu_util"], 1)
            sim_hw.racks[hw_chosen]["mem_available_pct"] = round(100.0 - res["new_mem_util"], 1)
            sim_hw.racks[hw_chosen].setdefault("active_workloads", []).append({
                "gpu_pct": wl.gpu_demand_pct, "mem_pct": wl.memory_demand_pct, "power_w": wl.power_est_w,
                "duration_remaining": max(1, wl.duration_s // 15)
            })

        # Advance time by 1 tick (15s)
        sim_rr.advance_time(1)
        sim_ll.advance_time(1)
        sim_hw.advance_time(1)

    # Compute final metrics
    for name, sim_inst in [("Round Robin", sim_rr), ("Least Loaded", sim_ll), ("HeatWise", sim_hw)]:
        temps = [r["temperature"] for r in sim_inst.racks.values()]
        powers = [r["total_power_w"] for r in sim_inst.racks.values()]
        peak_t = max(temps)
        hotspots = sum(1 for t in temps if t >= config.TEMP_CRITICAL)
        sla_viols = sum(1 for t in temps if t >= config.TEMP_MAX_SAFE) + hotspots
        total_energy_kwh = (sum(powers) / 1000.0) * (num_workloads * 0.25)
        avg_headroom = float(np.mean([max(0.0, config.TEMP_CRITICAL - t) for t in temps]))

        metrics[name].update({
            "peak_temp": round(peak_t, 1),
            "avg_temp": round(float(np.mean(temps)), 1),
            "temp_std": round(float(np.std(temps)), 2),
            "hotspots": hotspots,
            "sla_violations": sla_viols,
            "avg_headroom": round(avg_headroom, 1),
            "total_power_kw": round(sum(powers) / 1000.0, 2),
            "total_energy_kwh": round(total_energy_kwh, 2)
        })

    return metrics


# ==============================================================================
# EXPERIMENT H: MULTI-SEED ROBUSTNESS (SEEDS 100, 200, 300, 400, 500)
# ==============================================================================
def run_experiment_h_multiseed() -> Dict[str, Any]:
    print("\n=======================================================")
    print("--- [EXP H] Running Multi-Seed Robustness Benchmark ---")
    print("=======================================================")
    seeds = [100, 200, 300, 400, 500]
    num_workloads = 15

    seed_records = []

    for seed in seeds:
        np.random.seed(seed)
        workload_types = ["AI Training", "AI Inference", "LLM Inference", "Image Processing"]
        intensities = ["Low", "Medium", "High", "Very High"]
        priorities = ["Medium", "High", "Critical"]

        stream = []
        for i in range(1, num_workloads + 1):
            w_type = np.random.choice(workload_types, p=[0.35, 0.30, 0.25, 0.10])
            w_int = np.random.choice(intensities, p=[0.20, 0.40, 0.30, 0.10])
            w_pri = np.random.choice(priorities, p=[0.30, 0.50, 0.20])
            stream.append(heatwise.Workload(
                workload_type=w_type,
                intensity=w_int,
                priority=w_pri,
                workload_id=f"SEED-{seed}-JOB-{100 + i}"
            ))

        res = run_single_simulation_stream(stream, base_scenario="Thermal Hotspot")
        for sched in ["Round Robin", "Least Loaded", "HeatWise"]:
            rec = {"seed": seed, "scheduler": sched}
            rec.update(res[sched])
            seed_records.append(rec)

    df_seeds = pd.DataFrame(seed_records)
    df_seeds.to_csv(os.path.join(RESULTS_DIR, "multiseed_benchmark.csv"), index=False)

    # Compute mean and standard deviation
    summary = {}
    for sched in ["Round Robin", "Least Loaded", "HeatWise"]:
        sub = df_seeds[df_seeds["scheduler"] == sched]
        summary[sched] = {
            "peak_temp_mean": round(float(sub["peak_temp"].mean()), 2),
            "peak_temp_std": round(float(sub["peak_temp"].std()), 2),
            "hotspots_mean": round(float(sub["hotspots"].mean()), 2),
            "hotspots_std": round(float(sub["hotspots"].std()), 2),
            "sla_violations_mean": round(float(sub["sla_violations"].mean()), 2),
            "sla_violations_std": round(float(sub["sla_violations"].std()), 2),
            "temp_std_mean": round(float(sub["temp_std"].mean()), 2),
            "temp_std_std": round(float(sub["temp_std"].std()), 2),
            "avg_headroom_mean": round(float(sub["avg_headroom"].mean()), 2),
            "avg_headroom_std": round(float(sub["avg_headroom"].std()), 2),
            "total_power_kw_mean": round(float(sub["total_power_kw"].mean()), 2),
            "total_energy_kwh_mean": round(float(sub["total_energy_kwh"].mean()), 2)
        }

    with open(os.path.join(RESULTS_DIR, "multiseed_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for k, v in summary.items():
        print(f"{k}: Peak={v['peak_temp_mean']}±{v['peak_temp_std']}°C, Hotspots={v['hotspots_mean']}±{v['hotspots_std']}, SLA={v['sla_violations_mean']}±{v['sla_violations_std']}, TempStd={v['temp_std_mean']}±{v['temp_std_std']}°C")

    return summary


# ==============================================================================
# EXPERIMENT I: BURSTY ARRIVAL STRESS TEST (POISSON LAMBDA=2.5)
# ==============================================================================
def run_experiment_i_bursty() -> Dict[str, Any]:
    print("\n=======================================================")
    print("--- [EXP I] Running Bursty Arrival Stress Test ---")
    print("=======================================================")
    np.random.seed(42)
    # Poisson arrivals with lambda=2.5 across 12 arrival windows
    # Total jobs ~ 30
    windows = 12
    stream = []
    job_idx = 1
    for w in range(windows):
        n_arrivals = int(np.random.poisson(lam=2.5))
        for _ in range(n_arrivals):
            # 80% High/Very High AI Training + LLM Inference, 20% Medium
            is_heavy = np.random.rand() < 0.80
            if is_heavy:
                w_type = np.random.choice(["AI Training", "LLM Inference"])
                w_int = np.random.choice(["High", "Very High"])
            else:
                w_type = np.random.choice(["AI Inference", "Image Processing"])
                w_int = "Medium"

            stream.append(heatwise.Workload(
                workload_type=w_type,
                intensity=w_int,
                priority="High",
                workload_id=f"BURST-{job_idx}"
            ))
            job_idx += 1

    print(f"Generated bursty stream with {len(stream)} rapid arrivals.")
    res = run_single_simulation_stream(stream, base_scenario="Thermal Hotspot")

    records = []
    for sched, m in res.items():
        rec = {"experiment": "Bursty Stress", "scheduler": sched, "total_workloads": len(stream)}
        rec.update(m)
        records.append(rec)

    df_burst = pd.DataFrame(records)
    df_burst.to_csv(os.path.join(RESULTS_DIR, "bursty_stress.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "bursty_stress.json"), "w") as f:
        json.dump(records, f, indent=2)

    for r in records:
        print(f"{r['scheduler']}: Peak={r['peak_temp']}°C, Hotspots={r['hotspots']}, SLA={r['sla_violations']}, TempStd={r['temp_std']}°C, Failures={r['placement_failures']}")

    return {r["scheduler"]: r for r in records}


# ==============================================================================
# EXPERIMENT J: HIGH-CONCURRENCY STRESS TEST (50 WORKLOADS)
# ==============================================================================
def run_experiment_j_concurrency() -> Dict[str, Any]:
    print("\n=======================================================")
    print("--- [EXP J] Running High-Concurrency Stress Test ---")
    print("=======================================================")
    np.random.seed(1337)
    num_workloads = 50
    stream = []
    for i in range(1, num_workloads + 1):
        w_type = np.random.choice(["AI Training", "AI Inference", "LLM Inference", "Image Processing"])
        w_int = np.random.choice(["Low", "Medium", "High", "Very High"])
        # Custom duration 15s to 90s
        dur = int(np.random.choice([15, 30, 45, 60, 75, 90]))
        wl = heatwise.Workload(
            workload_type=w_type,
            intensity=w_int,
            priority="High",
            workload_id=f"CONCUR-{i}"
        )
        wl.duration_s = dur
        stream.append(wl)

    res = run_single_simulation_stream(stream, base_scenario="Thermal Hotspot")
    records = []
    for sched, m in res.items():
        rec = {"experiment": "Concurrency Stress", "scheduler": sched, "total_workloads": num_workloads}
        rec.update(m)
        records.append(rec)

    df_concur = pd.DataFrame(records)
    df_concur.to_csv(os.path.join(RESULTS_DIR, "concurrency_stress.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "concurrency_stress.json"), "w") as f:
        json.dump(records, f, indent=2)

    for r in records:
        print(f"{r['scheduler']}: Peak={r['peak_temp']}°C, Hotspots={r['hotspots']}, SLA={r['sla_violations']}, Failures={r['placement_failures']}, FallbackInvocations={r['fallback_invocations']}")

    return {r["scheduler"]: r for r in records}


# ==============================================================================
# EXPERIMENT K: SUSTAINED HIGH-UTILIZATION REGIME (60 TIMESTEPS, >=85% LOAD)
# ==============================================================================
def run_experiment_k_sustained() -> Dict[str, Any]:
    print("\n=======================================================")
    print("--- [EXP K] Running Sustained High-Utilization Stress ---")
    print("=======================================================")
    np.random.seed(2026)
    # Generate 60 heavy workloads with long duration to keep load >=85%
    timesteps = 60
    stream = []
    for t in range(1, timesteps + 1):
        # Heavy AI Training and LLM jobs with 300s duration
        w_type = np.random.choice(["AI Training", "LLM Inference"], p=[0.7, 0.3])
        wl = heatwise.Workload(
            workload_type=w_type,
            intensity="High",
            priority="Critical",
            workload_id=f"SUSTAIN-{t}"
        )
        wl.duration_s = 300 # 20 ticks persistence
        stream.append(wl)

    res = run_single_simulation_stream(stream, base_scenario="High AI Workload")
    records = []
    for sched, m in res.items():
        rec = {"experiment": "Sustained High-Load", "scheduler": sched, "timesteps": timesteps}
        rec.update(m)
        records.append(rec)

    df_sustain = pd.DataFrame(records)
    df_sustain.to_csv(os.path.join(RESULTS_DIR, "sustained_load.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "sustained_load.json"), "w") as f:
        json.dump(records, f, indent=2)

    for r in records:
        print(f"{r['scheduler']}: Peak={r['peak_temp']}°C, Avg={r['avg_temp']}°C, TempStd={r['temp_std']}°C, Hotspots={r['hotspots']}, SLA={r['sla_violations']}, Failures={r['placement_failures']}")

    return {r["scheduler"]: r for r in records}


if __name__ == "__main__":
    run_experiment_h_multiseed()
    run_experiment_i_bursty()
    run_experiment_j_concurrency()
    run_experiment_k_sustained()
