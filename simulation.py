"""
HeatWise: Data Center Physical Simulation Module
Simulates a 12-rack (3x4 grid) high-density AI data center with 48 GPU servers.
Models realistic relationships between GPU/CPU utilization, power consumption,
cooling efficiency, spatial neighboring heat recirculation, and equilibrium temperatures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from config import (
    GRID_ROWS, GRID_COLS, TOTAL_RACKS, SERVERS_PER_RACK, GPUS_PER_SERVER,
    TEMP_NORMAL_MAX, TEMP_WARM_MAX, TEMP_HIGH_MAX, TEMP_CRITICAL, TEMP_MAX_SAFE,
    AMBIENT_TEMP, SERVER_IDLE_POWER_W, SERVER_MAX_CPU_POWER_W, SERVER_MAX_GPU_POWER_W,
    SIMULATION_RANDOM_SEED, TELEMETRY_RANDOM_SEED, SERVER_MAX_MEMORY_GB
)

# Rack coordinate mapping: rack_id (1-12) -> (row, col)
RACK_COORDS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
    5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
    9: (2, 0), 10: (2, 1), 11: (2, 2), 12: (2, 3)
}
COORDS_TO_RACK = {v: k for k, v in RACK_COORDS.items()}


def get_neighbor_rack_ids(rack_id: int) -> List[int]:
    """Returns direct orthogonal neighbors in the 3x4 grid (Manhattan distance = 1)."""
    r, c = RACK_COORDS[rack_id]
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            neighbors.append(COORDS_TO_RACK[(nr, nc)])
    return neighbors


def compute_server_power(cpu_util: float, gpu_util: float) -> float:
    """Computes realistic server power consumption in Watts."""
    cpu_p = (cpu_util / 100.0) * SERVER_MAX_CPU_POWER_W
    gpu_p = (gpu_util / 100.0) * SERVER_MAX_GPU_POWER_W
    return round(SERVER_IDLE_POWER_W + cpu_p + gpu_p, 1)


def compute_equilibrium_temperature(
    total_power_w: float,
    cooling_capacity_pct: float,
    neighbor_avg_temp: float,
    recirculation_factor: float = 1.0
) -> Tuple[float, float, float]:
    """
    Computes thermodynamic equilibrium temperature (°C) for a rack.
    Returns: (inlet_temp, outlet_temp, equilibrium_temp)
    """
    power_kw = total_power_w / 1000.0
    
    # Inlet temp is influenced by ambient and neighbor recirculation
    neighbor_diff = max(0.0, neighbor_avg_temp - AMBIENT_TEMP)
    neighbor_influence = 0.14 * neighbor_diff
    inlet_temp = AMBIENT_TEMP + neighbor_influence

    # Power temperature rise: ~3.8°C per kW of heat generated
    raw_heat_rise = power_kw * 3.85 * recirculation_factor
    
    # Cooling dampening: higher cooling capacity extracts heat more effectively
    cooling_factor = max(0.20, (cooling_capacity_pct / 100.0))
    effective_heat_rise = raw_heat_rise / (cooling_factor ** 0.85)
    
    outlet_temp = inlet_temp + raw_heat_rise
    final_temp = inlet_temp + effective_heat_rise
    
    return (
        round(float(np.clip(inlet_temp, 18.0, 50.0)), 1),
        round(float(np.clip(outlet_temp, 25.0, 100.0)), 1),
        round(float(np.clip(final_temp, 28.0, 92.0)), 1)
    )


def get_thermal_status(temp: float) -> Tuple[str, str, str]:
    """Returns (status_label, hex_color, badge_class)."""
    if temp < TEMP_NORMAL_MAX:
        return "Normal", "#10b981", "status-normal"
    elif temp < TEMP_WARM_MAX:
        return "Warm", "#f59e0b", "status-warm"
    elif temp < TEMP_CRITICAL:
        return "High", "#f97316", "status-high"
    else:
        return "Hotspot", "#ef4444", "status-hotspot"


class DataCenterSimulation:
    """Main Data Center Simulation Model with 12 Racks and 48 GPU Servers."""

    def __init__(self, scenario: str = "Thermal Hotspot"):
        self.scenario = scenario
        self.racks: Dict[int, Dict[str, Any]] = {}
        self.load_scenario(scenario)

    def load_scenario(self, scenario_name: str):
        """Initializes the data center state based on the selected scenario."""
        self.scenario = scenario_name
        self.racks = {}

        # Default baseline seed values for 12 racks
        np.random.seed(SIMULATION_RANDOM_SEED)

        # Baseline profiles (CPU %, GPU %, Memory %, Cooling %, Recirculation Factor)
        base_configs = {
            1:  {"cpu": 45, "gpu": 50, "mem": 60, "cooling": 92, "recirc": 1.00, "workload": "BERT Embeddings"},
            2:  {"cpu": 55, "gpu": 60, "mem": 65, "cooling": 90, "recirc": 1.02, "workload": "ResNet Vision"},
            3:  {"cpu": 35, "gpu": 40, "mem": 45, "cooling": 96, "recirc": 0.95, "workload": "Light Inference"},
            4:  {"cpu": 60, "gpu": 65, "mem": 70, "cooling": 88, "recirc": 1.00, "workload": "Audio Transcription"},
            5:  {"cpu": 50, "gpu": 55, "mem": 60, "cooling": 90, "recirc": 1.05, "workload": "Recommendation Sys"},
            6:  {"cpu": 65, "gpu": 70, "mem": 75, "cooling": 85, "recirc": 1.08, "workload": "Diffusion Model"},
            7:  {"cpu": 25, "gpu": 30, "mem": 35, "cooling": 88, "recirc": 1.00, "workload": "Idle / Standby"},
            8:  {"cpu": 50, "gpu": 58, "mem": 65, "cooling": 90, "recirc": 1.00, "workload": "NLP Tokenizer"},
            9:  {"cpu": 40, "gpu": 45, "mem": 50, "cooling": 94, "recirc": 0.96, "workload": "Vector Search"},
            10: {"cpu": 60, "gpu": 68, "mem": 75, "cooling": 86, "recirc": 1.02, "workload": "RL Agent Training"},
            11: {"cpu": 45, "gpu": 52, "mem": 60, "cooling": 92, "recirc": 0.98, "workload": "Graph Neural Net"},
            12: {"cpu": 55, "gpu": 62, "mem": 70, "cooling": 88, "recirc": 1.00, "workload": "Video Analytics"},
        }

        if scenario_name == "Thermal Hotspot":
            # Scenario: Rack 7 has high GPU availability (70% free, 30% util)
            # but due to hot-aisle air trap & degraded localized cooling, it is already thermally stressed (68.2°C).
            # Rack 3 has 60% free GPU (40% util) with 96% cooling capacity and safe 48.1°C temp.
            base_configs[7] = {"cpu": 25, "gpu": 30, "mem": 35, "cooling": 64, "recirc": 1.38, "workload": "Batch Worker"}
            base_configs[6] = {"cpu": 75, "gpu": 82, "mem": 85, "cooling": 80, "recirc": 1.15, "workload": "LLM Fine-tuning"}
            base_configs[8] = {"cpu": 70, "gpu": 78, "mem": 80, "cooling": 82, "recirc": 1.12, "workload": "Dense Retrieval"}
            base_configs[3] = {"cpu": 35, "gpu": 40, "mem": 45, "cooling": 96, "recirc": 0.92, "workload": "API Gateway"}
            base_configs[2] = {"cpu": 40, "gpu": 45, "mem": 50, "cooling": 94, "recirc": 0.95, "workload": "Inference Cache"}

        elif scenario_name == "Normal Operation":
            # Well distributed moderate temperatures
            for r_id in base_configs:
                base_configs[r_id]["cooling"] = 92
                base_configs[r_id]["recirc"] = 0.98

        elif scenario_name == "High AI Workload":
            # Cluster zone in Top-Left (Racks 1, 2, 5, 6) under heavy load
            for r_id in [1, 2, 5, 6, 10]:
                base_configs[r_id]["gpu"] = min(92, base_configs[r_id]["gpu"] + 25)
                base_configs[r_id]["cpu"] = min(88, base_configs[r_id]["cpu"] + 20)
                base_configs[r_id]["recirc"] = 1.18

        elif scenario_name == "Cooling Degradation":
            # Middle row CRAC airflow failure: Racks 5, 6, 7, 8 have cooling drop to 42-48%
            for r_id in [5, 6, 7, 8]:
                base_configs[r_id]["cooling"] = 45
                base_configs[r_id]["recirc"] = 1.25

        elif scenario_name == "GPU Surge":
            # Sudden surge across bottom row (Racks 9, 10, 11, 12)
            for r_id in [9, 10, 11, 12]:
                base_configs[r_id]["gpu"] = 88
                base_configs[r_id]["cpu"] = 80

        # Build detailed server & rack data
        temp_estimates = {}
        for r_id, cfg in base_configs.items():
            servers = []
            for s_idx in range(1, SERVERS_PER_RACK + 1):
                # Small per-server variance (+- 4%)
                s_cpu = float(np.clip(cfg["cpu"] + np.random.uniform(-4, 4), 10, 98))
                s_gpu = float(np.clip(cfg["gpu"] + np.random.uniform(-5, 5), 10, 98))
                s_mem = float(np.clip(cfg.get("mem", 50) + np.random.uniform(-5, 5), 10, 98))
                s_power = compute_server_power(s_cpu, s_gpu)
                servers.append({
                    "server_id": f"R{r_id}-S{s_idx}",
                    "server_num": s_idx,
                    "cpu_util": round(s_cpu, 1),
                    "gpu_util": round(s_gpu, 1),
                    "mem_util": round(s_mem, 1),
                    "gpu_count": GPUS_PER_SERVER,
                    "mem_capacity_gb": SERVER_MAX_MEMORY_GB,
                    "power_w": s_power,
                    "active_workload": cfg["workload"]
                })
            
            total_rack_power = sum(s["power_w"] for s in servers)
            avg_cpu = float(np.mean([s["cpu_util"] for s in servers]))
            avg_gpu = float(np.mean([s["gpu_util"] for s in servers]))
            avg_mem = float(np.mean([s["mem_util"] for s in servers]))
            total_mem_capacity = SERVERS_PER_RACK * SERVER_MAX_MEMORY_GB
            
            self.racks[r_id] = {
                "rack_id": r_id,
                "row": RACK_COORDS[r_id][0],
                "col": RACK_COORDS[r_id][1],
                "servers": servers,
                "total_power_w": round(total_rack_power, 1),
                "avg_cpu_util": round(avg_cpu, 1),
                "avg_gpu_util": round(avg_gpu, 1),
                "avg_mem_util": round(avg_mem, 1),
                "gpu_available_pct": round(100.0 - avg_gpu, 1),
                "gpu_available_count": int(round((100.0 - avg_gpu) / 100.0 * (SERVERS_PER_RACK * GPUS_PER_SERVER))),
                "mem_available_pct": round(100.0 - avg_mem, 1),
                "mem_available_gb": round((100.0 - avg_mem) / 100.0 * total_mem_capacity, 1),
                "mem_capacity_gb": total_mem_capacity,
                "cooling_capacity_pct": cfg["cooling"],
                "recirculation_factor": cfg["recirc"],
                "primary_workload": cfg["workload"],
                "temperature": 45.0,  # Placeholder, computed next
                "inlet_temp": 22.0,
                "outlet_temp": 50.0,
                "active_workloads": []  # List of dicts with workload info and remaining duration
            }
            temp_estimates[r_id] = 45.0

        # Iterative thermal equilibrium solver (incorporating neighbors)
        for _ in range(4):
            for r_id in range(1, TOTAL_RACKS + 1):
                nbr_ids = get_neighbor_rack_ids(r_id)
                nbr_avg = float(np.mean([temp_estimates[nid] for nid in nbr_ids]))
                inlet, outlet, t_eq = compute_equilibrium_temperature(
                    self.racks[r_id]["total_power_w"],
                    self.racks[r_id]["cooling_capacity_pct"],
                    nbr_avg,
                    self.racks[r_id]["recirculation_factor"]
                )
                temp_estimates[r_id] = t_eq
                self.racks[r_id]["inlet_temp"] = inlet
                self.racks[r_id]["outlet_temp"] = outlet

        # Finalize rack properties
        for r_id in range(1, TOTAL_RACKS + 1):
            nbr_ids = get_neighbor_rack_ids(r_id)
            nbr_avg = round(float(np.mean([temp_estimates[nid] for nid in nbr_ids])), 1)
            temp = temp_estimates[r_id]
            status, color, badge = get_thermal_status(temp)
            headroom = round(max(0.0, TEMP_CRITICAL - temp), 1)

            self.racks[r_id]["temperature"] = temp
            self.racks[r_id]["neighbor_avg_temp"] = nbr_avg
            self.racks[r_id]["neighbor_ids"] = nbr_ids
            self.racks[r_id]["thermal_headroom"] = headroom
            self.racks[r_id]["status"] = status
            self.racks[r_id]["status_color"] = color
            self.racks[r_id]["badge_class"] = badge

    def get_rack_df(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame of all rack metrics."""
        records = []
        for r_id in range(1, TOTAL_RACKS + 1):
            r = self.racks[r_id]
            records.append({
                "Rack ID": f"Rack {r_id}",
                "Rack Number": r_id,
                "Row": r["row"] + 1,
                "Col": r["col"] + 1,
                "Temperature (°C)": r["temperature"],
                "Inlet Temp (°C)": r["inlet_temp"],
                "Outlet Temp (°C)": r["outlet_temp"],
                "Status": r["status"],
                "Thermal Headroom (°C)": r["thermal_headroom"],
                "GPU Available (%)": r["gpu_available_pct"],
                "GPU Count Free": r["gpu_available_count"],
                "Mem Available (%)": r["mem_available_pct"],
                "Avg GPU Util (%)": r["avg_gpu_util"],
                "Avg CPU Util (%)": r["avg_cpu_util"],
                "Avg Mem Util (%)": r["avg_mem_util"],
                "Total Power (kW)": round(r["total_power_w"] / 1000.0, 2),
                "Cooling Capacity (%)": r["cooling_capacity_pct"],
                "Neighbor Avg Temp (°C)": r["neighbor_avg_temp"],
                "Primary Workload": r["primary_workload"]
            })
        return pd.DataFrame(records)

    def get_kpi_summary(self) -> Dict[str, Any]:
        """Calculates top-level KPI metrics for the entire data center."""
        temps = [r["temperature"] for r in self.racks.values()]
        powers = [r["total_power_w"] for r in self.racks.values()]
        gpu_utils = [r["avg_gpu_util"] for r in self.racks.values()]
        hotspots = sum(1 for t in temps if t >= TEMP_CRITICAL)
        warm_count = sum(1 for t in temps if TEMP_WARM_MAX <= t < TEMP_CRITICAL)
        headrooms = [r["thermal_headroom"] for r in self.racks.values()]
        active_workloads = sum(len(r.get("active_workloads", [])) for r in self.racks.values())
        if active_workloads == 0:
            active_workloads = sum(1 for r in self.racks.values() if r["avg_gpu_util"] > 10) * SERVERS_PER_RACK

        return {
            "total_racks": TOTAL_RACKS,
            "total_servers": TOTAL_RACKS * SERVERS_PER_RACK,
            "total_gpus": TOTAL_RACKS * SERVERS_PER_RACK * GPUS_PER_SERVER,
            "active_workloads": active_workloads,
            "avg_temp": round(float(np.mean(temps)), 1),
            "max_temp": round(float(np.max(temps)), 1),
            "min_temp": round(float(np.min(temps)), 1),
            "total_power_kw": round(float(np.sum(powers)) / 1000.0, 2),
            "avg_gpu_util": round(float(np.mean(gpu_utils)), 1),
            "hotspot_count": hotspots,
            "thermal_violations": hotspots,
            "warm_count": warm_count,
            "healthy_racks": TOTAL_RACKS - hotspots - warm_count,
            "avg_thermal_headroom": round(float(np.mean(headrooms)), 1),
            "placement_failures": getattr(self, 'placement_failures', 0)
        }

    def simulate_placement(self, rack_id: int, workload_gpu_pct: float, workload_mem_pct: float, workload_power_w: float) -> Dict[str, Any]:
        """
        Simulates the physical outcome of placing a workload on a given rack.
        Returns the new state without modifying the persistent simulation.
        """
        curr = self.racks[rack_id]
        new_power_w = curr["total_power_w"] + workload_power_w
        new_gpu_util = min(100.0, curr["avg_gpu_util"] + (workload_gpu_pct / SERVERS_PER_RACK))
        new_mem_util = min(100.0, curr["avg_mem_util"] + (workload_mem_pct / SERVERS_PER_RACK))
        
        # New temperature under modified power
        inlet, outlet, new_temp = compute_equilibrium_temperature(
            new_power_w,
            curr["cooling_capacity_pct"],
            curr["neighbor_avg_temp"],
            curr["recirculation_factor"]
        )
        new_headroom = max(0.0, TEMP_CRITICAL - new_temp)
        status, color, badge = get_thermal_status(new_temp)

        return {
            "rack_id": rack_id,
            "prev_temp": curr["temperature"],
            "new_temp": new_temp,
            "new_inlet_temp": inlet,
            "new_outlet_temp": outlet,
            "temp_delta": round(new_temp - curr["temperature"], 1),
            "prev_power_w": curr["total_power_w"],
            "new_power_w": round(new_power_w, 1),
            "new_headroom": round(new_headroom, 1),
            "new_status": status,
            "new_status_color": color,
            "new_gpu_util": round(new_gpu_util, 1),
            "new_mem_util": round(new_mem_util, 1),
            "is_hotspot": new_temp >= TEMP_CRITICAL,
            "sla_violation": new_temp >= TEMP_CRITICAL
        }

    def advance_time(self, steps: int = 1):
        """Advances simulation time, decrements workload duration, and releases resources."""
        for _ in range(steps):
            for r_id, rack in self.racks.items():
                active_workloads = rack.get("active_workloads", [])
                still_active = []
                for wl in active_workloads:
                    wl["duration_remaining"] -= 1
                    if wl["duration_remaining"] > 0:
                        still_active.append(wl)
                    else:
                        # Release resources
                        rack["total_power_w"] = max(400.0 * SERVERS_PER_RACK, rack["total_power_w"] - wl["power_w"])
                        rack["avg_gpu_util"] = max(0.0, rack["avg_gpu_util"] - (wl["gpu_pct"] / SERVERS_PER_RACK))
                        rack["avg_mem_util"] = max(0.0, rack["avg_mem_util"] - (wl["mem_pct"] / SERVERS_PER_RACK))
                
                rack["active_workloads"] = still_active
                rack["gpu_available_pct"] = round(100.0 - rack["avg_gpu_util"], 1)
                rack["mem_available_pct"] = round(100.0 - rack["avg_mem_util"], 1)
                
        # Recompute temperatures after resources might have been released
        for r_id, rack in self.racks.items():
            inlet, outlet, t_eq = compute_equilibrium_temperature(
                rack["total_power_w"],
                rack["cooling_capacity_pct"],
                rack["neighbor_avg_temp"],
                rack["recirculation_factor"]
            )
            rack["temperature"] = t_eq
            rack["inlet_temp"] = inlet
            rack["outlet_temp"] = outlet
            rack["thermal_headroom"] = max(0.0, TEMP_CRITICAL - t_eq)
            status, color, badge = get_thermal_status(t_eq)
            rack["status"] = status
            rack["status_color"] = color
            rack["badge_class"] = badge


def generate_synthetic_training_data(n_samples: int = 3500) -> pd.DataFrame:
    """
    Generates realistic, physics-grounded synthetic telemetry for ML model training.
    Uses dynamic trajectory generation to cover the true operational state space.
    """
    np.random.seed(TELEMETRY_RANDOM_SEED)
    samples = []
    scenarios = ["Normal Operation", "GPU Surge", "Cooling Degradation", "Thermal Hotspot", "High AI Workload"]
    
    while len(samples) < n_samples:
        scenario = np.random.choice(scenarios)
        sim = DataCenterSimulation(scenario=scenario)
        
        # Run a trajectory of workloads
        for _ in range(60):
            if len(samples) >= n_samples:
                break
                
            wl_types = ["AI Training", "AI Inference", "LLM Inference", "Image Processing"]
            wl_type = np.random.choice(wl_types, p=[0.35, 0.30, 0.25, 0.10])
            w_int = np.random.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.30, 0.10])
            
            mult = 1.0 + (w_int - 1) * 0.25
            base_g, base_m, base_p = 50.0, 50.0, 1000
            if wl_type == "AI Training":
                base_g, base_m, base_p = 80.0, 70.0, 2000
            elif wl_type == "LLM Inference":
                base_g, base_m, base_p = 60.0, 85.0, 1500
            elif wl_type == "AI Inference":
                base_g, base_m, base_p = 40.0, 40.0, 800
            elif wl_type == "Image Processing":
                base_g, base_m, base_p = 20.0, 30.0, 400
                
            gpu_pct = round(base_g * mult, 1)
            mem_pct = round(base_m * mult, 1)
            power_w = round(base_p * mult, 1)
            
            for r_id in range(1, 13):
                rack = sim.racks[r_id]
                
                feat = {
                    "cpu_utilization": rack["avg_cpu_util"],
                    "gpu_utilization": rack["avg_gpu_util"],
                    "memory_utilization": rack.get("avg_mem_util", 50.0),
                    "inlet_temperature": rack.get("inlet_temp", 22.0),
                    "local_power_kw": rack["total_power_w"] / 1000.0,
                    "neighbor_temperature": rack["neighbor_avg_temp"],
                    "neighbor_utilization": rack.get("neighbor_utilization", 50.0),
                    "cooling_capacity": rack["cooling_capacity_pct"],
                    "recirculation_factor": rack["recirculation_factor"],
                    
                    "workload_AI_Training": 1 if wl_type == "AI Training" else 0,
                    "workload_AI_Inference": 1 if wl_type == "AI Inference" else 0,
                    "workload_LLM": 1 if wl_type == "LLM Inference" else 0,
                    "workload_Image_Processing": 1 if wl_type == "Image Processing" else 0,
                    
                    "workload_intensity_val": w_int,
                    "workload_gpu_demand": gpu_pct,
                    "workload_mem_demand": mem_pct,
                    "workload_power_est": power_w
                }
                
                sim_res = sim.simulate_placement(r_id, gpu_pct, mem_pct, power_w)
                feat["post_workload_temp"] = sim_res["new_temp"]
                
                samples.append(feat)
                if len(samples) >= n_samples:
                    break
                    
            # Neutral / Randomized placement (relax constraint to explore extreme heat)
            valid_racks = list(range(1, 13))
            if valid_racks:
                winner = np.random.choice(valid_racks)
                sim.racks[winner]["total_power_w"] += power_w
                sim.racks[winner]["avg_gpu_util"] = min(100.0, sim.racks[winner]["avg_gpu_util"] + gpu_pct/SERVERS_PER_RACK)
                sim.racks[winner]["avg_mem_util"] = min(100.0, sim.racks[winner]["avg_mem_util"] + mem_pct/SERVERS_PER_RACK)
                sim.racks[winner]["gpu_available_pct"] = max(0.0, sim.racks[winner]["gpu_available_pct"] - gpu_pct/SERVERS_PER_RACK)
                sim.racks[winner]["mem_available_pct"] = max(0.0, sim.racks[winner]["mem_available_pct"] - mem_pct/SERVERS_PER_RACK)
                
                for _ in range(4):
                    temp_ests = {rid: r["temperature"] for rid, r in sim.racks.items()}
                    for rid in range(1, 13):
                        r = sim.racks[rid]
                        nbr = [temp_ests[nid] for nid in r["neighbor_ids"]]
                        inlet, outlet, t_eq = compute_equilibrium_temperature(r["total_power_w"], r["cooling_capacity_pct"], np.mean(nbr), r["recirculation_factor"])
                        sim.racks[rid]["temperature"] = t_eq
                        sim.racks[rid]["inlet_temp"] = inlet
                        sim.racks[rid]["neighbor_avg_temp"] = np.mean(nbr)
                        
    return pd.DataFrame(samples)
