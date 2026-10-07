"""
Experiments: Benchmark RNG Isolation Audit
Evaluates:
  Condition A: LEGACY CONTAMINATED BENCHMARK
               seed=100 -> train 3.5k model -> generate workloads -> run benchmark
  Condition B: MIGRATED UNISOLATED BENCHMARK
               seed=100 -> load 10k artifact -> generate workloads -> run benchmark
  Condition C: NORMALIZED ISOLATED BENCHMARK
               load model -> isolate RNG (seed=100) -> generate workloads -> run benchmark
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
import simulation
import ml_model
import heatwise


def run_benchmark_with_stream(workload_stream, predictor, base_scenario="Thermal Hotspot"):
    """Runs the exact benchmark logic on a provided workload stream using a given predictor."""
    engine = heatwise.HeatWiseDecisionEngine(predictor=predictor)

    sim_rr = simulation.DataCenterSimulation(scenario=base_scenario)
    sim_ll = simulation.DataCenterSimulation(scenario=base_scenario)
    sim_hw = simulation.DataCenterSimulation(scenario=base_scenario)

    metrics_tracker = {
        "Round Robin": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "placement_failures": 0},
        "Least Loaded": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "placement_failures": 0},
        "HeatWise": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "placement_failures": 0}
    }

    rr_idx = 1
    num_workloads = len(workload_stream)

    for step, wl in enumerate(workload_stream, 1):
        # 1. Round Robin Allocation
        rr_chosen_rack = None
        for _ in range(simulation.TOTAL_RACKS):
            candidate_id = rr_idx
            rr_idx = (rr_idx % simulation.TOTAL_RACKS) + 1
            rack = sim_rr.racks[candidate_id]
            if rack["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0):
                rr_chosen_rack = candidate_id
                break
        if rr_chosen_rack is None:
            rr_chosen_rack = 1

        rr_rack = sim_rr.racks[rr_chosen_rack]
        if rr_rack["mem_available_pct"] < (wl.memory_demand_pct / 4.0) or rr_rack["gpu_available_pct"] < (wl.gpu_demand_pct / 4.0):
            metrics_tracker["Round Robin"]["placement_failures"] += 1
        else:
            rr_res = sim_rr.simulate_placement(rr_chosen_rack, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_rr.racks[rr_chosen_rack]["total_power_w"] = rr_res["new_power_w"]
            sim_rr.racks[rr_chosen_rack]["avg_gpu_util"] = rr_res["new_gpu_util"]
            sim_rr.racks[rr_chosen_rack]["avg_mem_util"] = rr_res["new_mem_util"]
            sim_rr.racks[rr_chosen_rack]["temperature"] = rr_res["new_temp"]
            sim_rr.racks[rr_chosen_rack]["gpu_available_pct"] = round(100.0 - rr_res["new_gpu_util"], 1)
            sim_rr.racks[rr_chosen_rack]["mem_available_pct"] = round(100.0 - rr_res["new_mem_util"], 1)

        # 2. Least Loaded Allocation
        ll_candidates = [
            (r_id, r["gpu_available_pct"])
            for r_id, r in sim_ll.racks.items()
            if r["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0)
        ]
        if ll_candidates:
            ll_chosen_rack = max(ll_candidates, key=lambda x: x[1])[0]
        else:
            ll_chosen_rack = max(sim_ll.racks.items(), key=lambda x: x[1]["gpu_available_pct"])[0]

        ll_rack = sim_ll.racks[ll_chosen_rack]
        if ll_rack["mem_available_pct"] < (wl.memory_demand_pct / 4.0) or ll_rack["gpu_available_pct"] < (wl.gpu_demand_pct / 4.0):
            metrics_tracker["Least Loaded"]["placement_failures"] += 1
        else:
            ll_res = sim_ll.simulate_placement(ll_chosen_rack, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_ll.racks[ll_chosen_rack]["total_power_w"] = ll_res["new_power_w"]
            sim_ll.racks[ll_chosen_rack]["avg_gpu_util"] = ll_res["new_gpu_util"]
            sim_ll.racks[ll_chosen_rack]["avg_mem_util"] = ll_res["new_mem_util"]
            sim_ll.racks[ll_chosen_rack]["temperature"] = ll_res["new_temp"]
            sim_ll.racks[ll_chosen_rack]["gpu_available_pct"] = round(100.0 - ll_res["new_gpu_util"], 1)
            sim_ll.racks[ll_chosen_rack]["mem_available_pct"] = round(100.0 - ll_res["new_mem_util"], 1)

        # 3. HeatWise Allocation
        hw_eval = engine.evaluate_placement(sim_hw, wl)
        hw_chosen_rack = hw_eval["recommended_rack"]["rack_id"]

        if hw_eval["recommended_rack"]["is_rejected"]:
            metrics_tracker["HeatWise"]["placement_failures"] += 1
        else:
            hw_res = sim_hw.simulate_placement(hw_chosen_rack, wl.gpu_demand_pct, wl.memory_demand_pct, wl.power_est_w)
            sim_hw.racks[hw_chosen_rack]["total_power_w"] = hw_res["new_power_w"]
            sim_hw.racks[hw_chosen_rack]["avg_gpu_util"] = hw_res["new_gpu_util"]
            sim_hw.racks[hw_chosen_rack]["avg_mem_util"] = hw_res["new_mem_util"]
            sim_hw.racks[hw_chosen_rack]["temperature"] = hw_res["new_temp"]
            sim_hw.racks[hw_chosen_rack]["gpu_available_pct"] = round(100.0 - hw_res["new_gpu_util"], 1)
            sim_hw.racks[hw_chosen_rack]["mem_available_pct"] = round(100.0 - hw_res["new_mem_util"], 1)

        sim_rr.advance_time(1)
        sim_ll.advance_time(1)
        sim_hw.advance_time(1)

    summary = {}
    for name, sim_inst in [("Round Robin", sim_rr), ("Least Loaded", sim_ll), ("HeatWise", sim_hw)]:
        temps = [r["temperature"] for r in sim_inst.racks.values()]
        powers = [r["total_power_w"] for r in sim_inst.racks.values()]

        peak_t = max(temps)
        hotspots = sum(1 for t in temps if t >= simulation.TEMP_CRITICAL)
        sla_viols = sum(1 for t in temps if t >= simulation.TEMP_MAX_SAFE) + hotspots
        total_energy_kwh = (sum(powers) / 1000.0) * (num_workloads * 0.25)
        total_power_kw = sum(powers) / 1000.0
        headrooms = [simulation.TEMP_CRITICAL - t for t in temps]

        summary[name] = {
            "peak_temp": round(peak_t, 1),
            "temp_std": round(float(np.std(temps)), 1),
            "avg_temp": round(float(np.mean(temps)), 1),
            "avg_headroom": round(float(np.mean(headrooms)), 1),
            "hotspots": hotspots,
            "sla_violations": sla_viols,
            "placement_failures": metrics_tracker[name]["placement_failures"],
            "total_power_kw": round(total_power_kw, 2),
            "total_energy_kwh": round(total_energy_kwh, 2)
        }

    return summary


def generate_workload_sequence(rng, num_workloads=15):
    """Generates workload sequence using a provided RNG object or module."""
    workload_types = ["AI Training", "AI Inference", "LLM Inference", "Image Processing"]
    intensities = ["Low", "Medium", "High", "Very High"]
    priorities = ["Medium", "High", "Critical"]

    stream = []
    for i in range(1, num_workloads + 1):
        w_type = rng.choice(workload_types, p=[0.35, 0.30, 0.25, 0.10])
        w_int = rng.choice(intensities, p=[0.20, 0.40, 0.30, 0.10])
        w_pri = rng.choice(priorities, p=[0.30, 0.50, 0.20])
        stream.append(heatwise.Workload(
            workload_type=str(w_type),
            intensity=str(w_int),
            priority=str(w_pri),
            workload_id=f"JOB-{100 + i}"
        ))
    return stream


def run_audit():
    print("=" * 70)
    print("HEATWISE BENCHMARK RNG ISOLATION AUDIT")
    print("=" * 70)

    # 1. Condition A: LEGACY CONTAMINATED BENCHMARK
    print("\n--- 1. Condition A: LEGACY CONTAMINATED BENCHMARK ---")
    np.random.seed(100)
    legacy_pred = ml_model.ThermalMLPredictor(n_estimators=120, max_depth=14)
    # Explicitly run legacy 3,500 training which executes np.random.seed(42) and churns RNG
    legacy_pred.train()
    # Workloads generated using remaining global RNG state
    stream_A = generate_workload_sequence(np.random, 15)
    summary_A = run_benchmark_with_stream(stream_A, legacy_pred)

    # Also evaluate the 10k candidate model on Stream A to test Model effect on Stream A
    cand_pred = ml_model.get_trained_predictor()
    summary_A_with_10k = run_benchmark_with_stream(stream_A, cand_pred)

    # 2. Condition B: MIGRATED UNISOLATED BENCHMARK
    print("\n--- 2. Condition B: MIGRATED UNISOLATED BENCHMARK ---")
    np.random.seed(100)
    # Model loaded from artifact (0 RNG consumption)
    pred_B = ml_model.get_trained_predictor()
    stream_B = generate_workload_sequence(np.random, 15)
    summary_B = run_benchmark_with_stream(stream_B, pred_B)

    # 3. Condition C: NORMALIZED ISOLATED BENCHMARK
    print("\n--- 3. Condition C: NORMALIZED ISOLATED BENCHMARK ---")
    # Independent RNG instance seeded with 100
    rng_C = np.random.RandomState(100)
    stream_C = generate_workload_sequence(rng_C, 15)
    pred_C = ml_model.get_trained_predictor()
    summary_C = run_benchmark_with_stream(stream_C, pred_C)

    # Output detailed workload comparison
    print("\nWORKLOAD COMPARISON (Stream A vs Stream B vs Stream C):")
    stream_data = []
    for i in range(15):
        wA = stream_A[i]
        wB = stream_B[i]
        wC = stream_C[i]
        stream_data.append({
            "Job": wA.workload_id,
            "Stream A (Legacy)": f"{wA.workload_type} ({wA.intensity}) - {wA.power_est_w:.0f}W",
            "Stream B (Migrated)": f"{wB.workload_type} ({wB.intensity}) - {wB.power_est_w:.0f}W",
            "Stream C (Normalized)": f"{wC.workload_type} ({wC.intensity}) - {wC.power_est_w:.0f}W",
        })

    df_streams = pd.DataFrame(stream_data)
    print(df_streams.to_string(index=False))

    power_A = sum(w.power_est_w for w in stream_A)
    power_B = sum(w.power_est_w for w in stream_B)
    power_C = sum(w.power_est_w for w in stream_C)
    print(f"\nTotal Stream Added Power: A = {power_A:.1f} W | B = {power_B:.1f} W | C = {power_C:.1f} W")
    print(f"Stream B == Stream C identity check: {[wB.workload_type == wC.workload_type and wB.intensity == wC.intensity for wB, wC in zip(stream_B, stream_C)] == [True]*15}")

    results = {
        "summary_A_legacy_model": summary_A,
        "summary_A_10k_model": summary_A_with_10k,
        "summary_B_migrated": summary_B,
        "summary_C_normalized": summary_C,
        "power_A": power_A,
        "power_B": power_B,
        "power_C": power_C
    }

    out_file = os.path.join(REPO_ROOT, "experiments", "results", "BENCHMARK_RNG_AUDIT.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit results to {out_file}")

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISONS:")
    print("=" * 70)
    print("\nA. LEGACY CONTAMINATED (Stream A, 3.5k model):")
    for k, v in summary_A.items():
        print(f"  {k}: {v}")

    print("\nA2. LEGACY STREAM WITH 10K MODEL (Stream A, 10k model):")
    for k, v in summary_A_with_10k.items():
        print(f"  {k}: {v}")

    print("\nB. MIGRATED UNISOLATED (Stream B, 10k model):")
    for k, v in summary_B.items():
        print(f"  {k}: {v}")

    print("\nC. NORMALIZED ISOLATED (Stream C, 10k model):")
    for k, v in summary_C.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_audit()
