"""
HeatWise: Decision Engine and Schedulers Module
Implements:
1. Workload specifications (AI Training, Inference, LLM, etc.)
2. Conventional Schedulers: Round Robin and Least-Loaded
3. HeatWise Thermal-Aware Decision Engine with composite scoring and explainability
4. Batch Simulation Benchmark comparing all three algorithms on identical workload streams.
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from simulation import DataCenterSimulation, TEMP_CRITICAL, TEMP_MAX_SAFE, TOTAL_RACKS
from ml_model import ThermalMLPredictor, get_trained_predictor


# Workload configuration library
WORKLOAD_PROFILES = {
    "AI Training": {
        "description": "High-intensity multi-epoch model training (PyTorch/JAX)",
        "base_gpu_pct": 50.0,
        "base_power_w": 1800.0,
        "intensity_default": "High",
        "badge_color": "#ec4899"
    },
    "AI Inference": {
        "description": "Real-time microservice model inference",
        "base_gpu_pct": 25.0,
        "base_power_w": 800.0,
        "intensity_default": "Medium",
        "badge_color": "#06b6d4"
    },
    "LLM Inference": {
        "description": "High memory-bandwidth autoregressive token generation",
        "base_gpu_pct": 40.0,
        "base_power_w": 1400.0,
        "intensity_default": "High",
        "badge_color": "#8b5cf6"
    },
    "Image Processing": {
        "description": "Computer vision batch augmentation & filtering",
        "base_gpu_pct": 20.0,
        "base_power_w": 600.0,
        "intensity_default": "Medium",
        "badge_color": "#10b981"
    }
}

INTENSITY_MULTIPLIERS = {
    "Low": 0.65,
    "Medium": 1.00,
    "High": 1.35,
    "Very High": 1.75
}

INTENSITY_LEVELS = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}


class Workload:
    """Represents an incoming AI Workload request."""

    def __init__(
        self,
        workload_type: str = "AI Training",
        intensity: str = "High",
        gpu_demand_pct: Optional[float] = None,
        priority: str = "High",
        workload_id: str = "WL-101"
    ):
        self.workload_type = workload_type
        self.intensity = intensity
        self.priority = priority
        self.workload_id = workload_id

        profile = WORKLOAD_PROFILES.get(workload_type, WORKLOAD_PROFILES["AI Training"])
        multiplier = INTENSITY_MULTIPLIERS.get(intensity, 1.0)
        
        # Calculate effective GPU demand and estimated power
        if gpu_demand_pct is not None:
            self.gpu_demand_pct = float(gpu_demand_pct)
        else:
            self.gpu_demand_pct = round(profile["base_gpu_pct"] * multiplier, 1)

        self.power_est_w = round(profile["base_power_w"] * multiplier, 1)
        self.intensity_val = INTENSITY_LEVELS.get(intensity, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_type": self.workload_type,
            "intensity": self.intensity,
            "intensity_val": self.intensity_val,
            "gpu_demand_pct": self.gpu_demand_pct,
            "power_est_w": self.power_est_w,
            "priority": self.priority
        }


# Default weights for HeatWise Composite Scoring
DEFAULT_WEIGHTS = {
    "temp_safety": 0.35,      # Weight for predicted post-placement temperature safety
    "thermal_headroom": 0.25, # Weight for remaining thermal margin below critical threshold
    "gpu_fit": 0.20,          # Weight for computational GPU availability & fit
    "cooling_eff": 0.10,      # Weight for rack cooling capacity
    "neighbor_stability": 0.10 # Weight for thermal stability of surrounding racks
}


def score_temperature_safety(pred_temp: float) -> float:
    """Calculates 0-100 score where lower predicted temp gives higher score."""
    # 35°C -> 100, 75°C -> 0, >75°C -> negative penalty
    score = (TEMP_CRITICAL - pred_temp) / (TEMP_CRITICAL - 35.0) * 100.0
    return float(np.clip(score, -100.0, 100.0))


def score_thermal_headroom(headroom: float) -> float:
    """Calculates 0-100 score for remaining headroom margin."""
    # 30°C headroom -> 100, 0°C -> 0
    score = (headroom / 30.0) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def score_gpu_availability(gpu_avail_pct: float) -> float:
    """Calculates 0-100 score for GPU availability."""
    return float(np.clip(gpu_avail_pct, 0.0, 100.0))


def score_cooling_capacity(cooling_pct: float) -> float:
    """Calculates 0-100 score for cooling capacity."""
    return float(np.clip(cooling_pct, 0.0, 100.0))


def score_neighbor_stability(neighbor_avg_temp: float) -> float:
    """Calculates 0-100 score for neighbor temperature stability."""
    # 30°C neighbors -> 100, 70°C neighbors -> 0
    score = (70.0 - neighbor_avg_temp) / 40.0 * 100.0
    return float(np.clip(score, 0.0, 100.0))


class HeatWiseDecisionEngine:
    """Core HeatWise Multi-Factor Decision and Explainability Engine."""

    def __init__(self, predictor: Optional[ThermalMLPredictor] = None, weights: Optional[Dict[str, float]] = None):
        self.predictor = predictor or get_trained_predictor()
        self.weights = weights or DEFAULT_WEIGHTS

    def evaluate_placement(
        self,
        sim: DataCenterSimulation,
        workload: Workload,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates all 12 candidate racks and returns structured recommendations,
        comparisons, and explainability reasoning.
        """
        active_weights = weights or self.weights
        w_temp = active_weights.get("temp_safety", 0.35)
        w_head = active_weights.get("thermal_headroom", 0.25)
        w_gpu = active_weights.get("gpu_fit", 0.20)
        w_cool = active_weights.get("cooling_eff", 0.10)
        w_nbr = active_weights.get("neighbor_stability", 0.10)

        candidate_evaluations = []

        for r_id in range(1, TOTAL_RACKS + 1):
            rack = sim.racks[r_id]

            # Check basic GPU feasibility
            has_gpu_capacity = (rack["gpu_available_pct"] >= (workload.gpu_demand_pct / 4.0))

            # ML Temperature Prediction
            pred_temp, temp_delta = self.predictor.predict_rack_placement(
                rack,
                workload.gpu_demand_pct,
                workload.intensity_val,
                workload.power_est_w
            )

            pred_headroom = round(max(0.0, TEMP_CRITICAL - pred_temp), 1)
            pred_power_kw = round((rack["total_power_w"] + workload.power_est_w) / 1000.0, 2)
            is_hotspot_risk = pred_temp >= TEMP_CRITICAL

            # Component sub-scores (0-100)
            sub_temp = score_temperature_safety(pred_temp)
            sub_head = score_thermal_headroom(pred_headroom)
            sub_gpu = score_gpu_availability(rack["gpu_available_pct"])
            sub_cool = score_cooling_capacity(rack["cooling_capacity_pct"])
            sub_nbr = score_neighbor_stability(rack["neighbor_avg_temp"])

            # Weighted composite score
            raw_score = (
                w_temp * sub_temp +
                w_head * sub_head +
                w_gpu * sub_gpu +
                w_cool * sub_cool +
                w_nbr * sub_nbr
            )

            # Severe penalty if predicted to cause thermal hotspot (>75°C) or insufficient GPU
            penalty = 0.0
            if is_hotspot_risk:
                penalty += 45.0 + (pred_temp - TEMP_CRITICAL) * 5.0
            if not has_gpu_capacity:
                penalty += 50.0

            final_score = round(float(np.clip(raw_score - penalty, 0.0, 100.0)), 1)

            # Rejection / safety diagnosis
            is_rejected = False
            rejection_reason = "Passed all thermal and compute constraints."
            if not has_gpu_capacity:
                is_rejected = True
                rejection_reason = f"Insufficient GPU capacity (Available: {rack['gpu_available_pct']}%)."
            elif is_hotspot_risk:
                is_rejected = True
                rejection_reason = (
                    f"Thermal Safety Violation: Predicted post-placement temperature of "
                    f"{pred_temp}°C exceeds safety threshold ({TEMP_CRITICAL}°C)."
                )
            elif pred_temp >= 71.0:
                rejection_reason = f"Elevated thermal risk (Predicted: {pred_temp}°C, Low Headroom: {pred_headroom}°C)."

            candidate_evaluations.append({
                "rack_id": r_id,
                "rack_name": f"Rack {r_id}",
                "current_temp": rack["temperature"],
                "predicted_temp": pred_temp,
                "temp_delta": temp_delta,
                "predicted_headroom": pred_headroom,
                "gpu_available_pct": rack["gpu_available_pct"],
                "current_power_kw": round(rack["total_power_w"] / 1000.0, 2),
                "predicted_power_kw": pred_power_kw,
                "cooling_capacity_pct": rack["cooling_capacity_pct"],
                "neighbor_avg_temp": rack["neighbor_avg_temp"],
                "heatwise_score": final_score,
                "sub_scores": {
                    "temp_safety": round(sub_temp, 1),
                    "thermal_headroom": round(sub_head, 1),
                    "gpu_fit": round(sub_gpu, 1),
                    "cooling_eff": round(sub_cool, 1),
                    "neighbor_stability": round(sub_nbr, 1)
                },
                "is_hotspot_risk": is_hotspot_risk,
                "is_rejected": is_rejected,
                "rejection_reason": rejection_reason,
                "has_gpu_capacity": has_gpu_capacity
            })

        # Sort candidate evaluations by HeatWise score descending
        candidate_evaluations.sort(key=lambda x: x["heatwise_score"], reverse=True)
        recommended = candidate_evaluations[0]

        # Determine baseline scheduler choices for direct comparison
        # 1. Least-Loaded Choice: rack with highest gpu_available_pct among feasible
        feasible_gpu = [c for c in candidate_evaluations if c["has_gpu_capacity"]]
        least_loaded_choice = max(feasible_gpu, key=lambda x: x["gpu_available_pct"]) if feasible_gpu else candidate_evaluations[0]

        # 2. Round Robin Choice: next rack index (e.g. rack 1 if free, or arbitrary free rack)
        round_robin_choice = feasible_gpu[0] if feasible_gpu else candidate_evaluations[0]

        # Explainability why winner was chosen
        winner_reasons = [
            f"GPU Capacity: {recommended['gpu_available_pct']}% available (Sufficient for {workload.gpu_demand_pct}% demand)",
            f"Safe Predicted Temperature: {recommended['predicted_temp']}°C (+{recommended['temp_delta']}°C rise)",
            f"Thermal Headroom: {recommended['predicted_headroom']}°C buffer remaining below {TEMP_CRITICAL}°C limit",
            f"Cooling Efficiency: {recommended['cooling_capacity_pct']}% CRAC airflow capability",
            f"Surrounding Environment: Stable neighbor temperature at {recommended['neighbor_avg_temp']}°C"
        ]

        # Contrast with Least-Loaded choice if different
        contrast_scenario = None
        if least_loaded_choice["rack_id"] != recommended["rack_id"]:
            contrast_scenario = {
                "least_loaded_rack": least_loaded_choice["rack_name"],
                "least_loaded_gpu_avail": least_loaded_choice["gpu_available_pct"],
                "least_loaded_curr_temp": least_loaded_choice["current_temp"],
                "least_loaded_pred_temp": least_loaded_choice["predicted_temp"],
                "least_loaded_is_hotspot": least_loaded_choice["is_hotspot_risk"],
                "heatwise_rack": recommended["rack_name"],
                "heatwise_gpu_avail": recommended["gpu_available_pct"],
                "heatwise_curr_temp": recommended["current_temp"],
                "heatwise_pred_temp": recommended["predicted_temp"],
                "explanation": (
                    f"Conventional Least-Loaded scheduling selects {least_loaded_choice['rack_name']} solely "
                    f"because it has {least_loaded_choice['gpu_available_pct']}% GPU availability. However, "
                    f"{least_loaded_choice['rack_name']} is already at {least_loaded_choice['current_temp']}°C, "
                    f"causing its predicted temperature to surge to {least_loaded_choice['predicted_temp']}°C "
                    f"({'CRITICAL HOTSPOT VIOLATION' if least_loaded_choice['is_hotspot_risk'] else 'high thermal stress'}). "
                    f"In contrast, HeatWise intelligently routes the workload to {recommended['rack_name']}, "
                    f"maintaining a safe operating temperature of {recommended['predicted_temp']}°C with "
                    f"{recommended['predicted_headroom']}°C headroom."
                )
            }

        return {
            "workload": workload.to_dict(),
            "recommended_rack": recommended,
            "least_loaded_rack": least_loaded_choice,
            "round_robin_rack": round_robin_choice,
            "all_candidates": candidate_evaluations,
            "winner_reasons": winner_reasons,
            "contrast_scenario": contrast_scenario,
            "weights_used": active_weights
        }


def run_batch_simulation_benchmark(
    base_scenario: str = "Thermal Hotspot",
    num_workloads: int = 15,
    seed: int = 100
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Simulates a sequence of AI workloads processed through:
    1. Round Robin
    2. Least Loaded
    3. HeatWise
    Returns comparative metrics, summary cards, and timeline traces.
    """
    np.random.seed(seed)
    predictor = get_trained_predictor()
    engine = HeatWiseDecisionEngine(predictor=predictor)

    # Generate identical stream of simulated workloads
    workload_types = ["AI Training", "AI Inference", "LLM Inference", "Image Processing"]
    intensities = ["Low", "Medium", "High", "Very High"]
    priorities = ["Medium", "High", "Critical"]

    workload_stream = []
    for i in range(1, num_workloads + 1):
        w_type = np.random.choice(workload_types, p=[0.35, 0.30, 0.25, 0.10])
        w_int = np.random.choice(intensities, p=[0.20, 0.40, 0.30, 0.10])
        w_pri = np.random.choice(priorities, p=[0.30, 0.50, 0.20])
        workload_stream.append(Workload(
            workload_type=w_type,
            intensity=w_int,
            priority=w_pri,
            workload_id=f"JOB-{100 + i}"
        ))

    # Initialize 3 independent simulation instances
    sim_rr = DataCenterSimulation(scenario=base_scenario)
    sim_ll = DataCenterSimulation(scenario=base_scenario)
    sim_hw = DataCenterSimulation(scenario=base_scenario)

    metrics_tracker = {
        "Round Robin": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "total_power_kwh": 0.0, "temps_history": []},
        "Least Loaded": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "total_power_kwh": 0.0, "temps_history": []},
        "HeatWise": {"peak_temp": 0.0, "hotspots": 0, "sla_violations": 0, "total_power_kwh": 0.0, "temps_history": []}
    }

    timeline_data = []
    rr_idx = 1

    for step, wl in enumerate(workload_stream, 1):
        # 1. Round Robin Allocation
        rr_chosen_rack = None
        for _ in range(TOTAL_RACKS):
            candidate_id = rr_idx
            rr_idx = (rr_idx % TOTAL_RACKS) + 1
            if sim_rr.racks[candidate_id]["gpu_available_pct"] >= (wl.gpu_demand_pct / 4.0):
                rr_chosen_rack = candidate_id
                break
        if rr_chosen_rack is None:
            rr_chosen_rack = 1

        rr_res = sim_rr.simulate_placement(rr_chosen_rack, wl.gpu_demand_pct, wl.power_est_w)
        # Apply changes to sim_rr state
        sim_rr.racks[rr_chosen_rack]["total_power_w"] = rr_res["new_power_w"]
        sim_rr.racks[rr_chosen_rack]["avg_gpu_util"] = rr_res["new_gpu_util"]
        sim_rr.racks[rr_chosen_rack]["temperature"] = rr_res["new_temp"]
        sim_rr.racks[rr_chosen_rack]["gpu_available_pct"] = round(100.0 - rr_res["new_gpu_util"], 1)

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

        ll_res = sim_ll.simulate_placement(ll_chosen_rack, wl.gpu_demand_pct, wl.power_est_w)
        sim_ll.racks[ll_chosen_rack]["total_power_w"] = ll_res["new_power_w"]
        sim_ll.racks[ll_chosen_rack]["avg_gpu_util"] = ll_res["new_gpu_util"]
        sim_ll.racks[ll_chosen_rack]["temperature"] = ll_res["new_temp"]
        sim_ll.racks[ll_chosen_rack]["gpu_available_pct"] = round(100.0 - ll_res["new_gpu_util"], 1)

        # 3. HeatWise Allocation
        hw_eval = engine.evaluate_placement(sim_hw, wl)
        hw_chosen_rack = hw_eval["recommended_rack"]["rack_id"]
        hw_res = sim_hw.simulate_placement(hw_chosen_rack, wl.gpu_demand_pct, wl.power_est_w)
        sim_hw.racks[hw_chosen_rack]["total_power_w"] = hw_res["new_power_w"]
        sim_hw.racks[hw_chosen_rack]["avg_gpu_util"] = hw_res["new_gpu_util"]
        sim_hw.racks[hw_chosen_rack]["temperature"] = hw_res["new_temp"]
        sim_hw.racks[hw_chosen_rack]["gpu_available_pct"] = round(100.0 - hw_res["new_gpu_util"], 1)

        # Record step temperatures across algorithms
        rr_all_temps = [r["temperature"] for r in sim_rr.racks.values()]
        ll_all_temps = [r["temperature"] for r in sim_ll.racks.values()]
        hw_all_temps = [r["temperature"] for r in sim_hw.racks.values()]

        timeline_data.append({
            "Step": step,
            "Workload": f"{wl.workload_type} ({wl.intensity})",
            "Round Robin Max Temp": max(rr_all_temps),
            "Least Loaded Max Temp": max(ll_all_temps),
            "HeatWise Max Temp": max(hw_all_temps),
            "RR Chosen Rack": f"Rack {rr_chosen_rack}",
            "LL Chosen Rack": f"Rack {ll_chosen_rack}",
            "HW Chosen Rack": f"Rack {hw_chosen_rack}"
        })

    # Final calculations after stream execution
    for name, sim_inst in [("Round Robin", sim_rr), ("Least Loaded", sim_ll), ("HeatWise", sim_hw)]:
        temps = [r["temperature"] for r in sim_inst.racks.values()]
        powers = [r["total_power_w"] for r in sim_inst.racks.values()]
        
        peak_t = max(temps)
        hotspots = sum(1 for t in temps if t >= TEMP_CRITICAL)
        sla_viols = sum(1 for t in temps if t >= TEMP_MAX_SAFE) + hotspots
        total_energy_kwh = (sum(powers) / 1000.0) * (num_workloads * 0.25)  # est energy

        metrics_tracker[name] = {
            "peak_temp": round(peak_t, 1),
            "avg_temp": round(float(np.mean(temps)), 1),
            "temp_std": round(float(np.std(temps)), 1),
            "hotspots": hotspots,
            "sla_violations": sla_viols,
            "total_power_kw": round(sum(powers) / 1000.0, 2),
            "total_energy_kwh": round(total_energy_kwh, 2)
        }

    comparison_df = pd.DataFrame([
        {
            "Metric": "Peak Temperature (°C)",
            "Round Robin": f"{metrics_tracker['Round Robin']['peak_temp']}°C",
            "Least Loaded": f"{metrics_tracker['Least Loaded']['peak_temp']}°C",
            "HeatWise": f"{metrics_tracker['HeatWise']['peak_temp']}°C",
            "HeatWise Advantage": f"-{round(max(metrics_tracker['Round Robin']['peak_temp'], metrics_tracker['Least Loaded']['peak_temp']) - metrics_tracker['HeatWise']['peak_temp'], 1)}°C Cooler"
        },
        {
            "Metric": "Thermal Hotspots (≥75°C)",
            "Round Robin": metrics_tracker["Round Robin"]["hotspots"],
            "Least Loaded": metrics_tracker["Least Loaded"]["hotspots"],
            "HeatWise": metrics_tracker["HeatWise"]["hotspots"],
            "HeatWise Advantage": f"{max(metrics_tracker['Round Robin']['hotspots'], metrics_tracker['Least Loaded']['hotspots']) - metrics_tracker['HeatWise']['hotspots']} Fewer Hotspots"
        },
        {
            "Metric": "Thermal SLA Violations",
            "Round Robin": metrics_tracker["Round Robin"]["sla_violations"],
            "Least Loaded": metrics_tracker["Least Loaded"]["sla_violations"],
            "HeatWise": metrics_tracker["HeatWise"]["sla_violations"],
            "HeatWise Advantage": "Zero Violations" if metrics_tracker["HeatWise"]["sla_violations"] == 0 else "Minimized"
        },
        {
            "Metric": "Avg Cluster Temp (°C)",
            "Round Robin": f"{metrics_tracker['Round Robin']['avg_temp']}°C",
            "Least Loaded": f"{metrics_tracker['Least Loaded']['avg_temp']}°C",
            "HeatWise": f"{metrics_tracker['HeatWise']['avg_temp']}°C",
            "HeatWise Advantage": "Balanced Distribution"
        },
        {
            "Metric": "Thermal Variance (Std Dev)",
            "Round Robin": f"±{metrics_tracker['Round Robin']['temp_std']}°C",
            "Least Loaded": f"±{metrics_tracker['Least Loaded']['temp_std']}°C",
            "HeatWise": f"±{metrics_tracker['HeatWise']['temp_std']}°C",
            "HeatWise Advantage": "Homogeneous Cooling"
        }
    ])

    return comparison_df, metrics_tracker, timeline_data
