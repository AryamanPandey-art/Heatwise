"""
HeatWise: Data Center Physical Simulation Module
Simulates a 12-rack (3x4 grid) high-density AI data center with 48 GPU servers.
Models realistic relationships between GPU/CPU utilization, power consumption,
cooling efficiency, spatial neighboring heat recirculation, and equilibrium temperatures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Data center topology: 3 rows x 4 columns = 12 racks
GRID_ROWS = 3
GRID_COLS = 4
TOTAL_RACKS = 12
SERVERS_PER_RACK = 4
GPUS_PER_SERVER = 8  # 8x GPUs (e.g., NVIDIA H100/A100 equivalent) = 32 GPUs per rack

# Thermal safety thresholds (°C)
TEMP_NORMAL_MAX = 60.0
TEMP_WARM_MAX = 68.0
TEMP_HIGH_MAX = 74.9
TEMP_CRITICAL = 75.0  # Threshold for thermal hotspot / SLA warning
TEMP_MAX_SAFE = 80.0  # Critical hardware throttle limit

# Physical constants for simulation
AMBIENT_TEMP = 22.0  # °C intake air baseline
SERVER_IDLE_POWER_W = 400.0  # Idle server power
SERVER_MAX_CPU_POWER_W = 350.0
SERVER_MAX_GPU_POWER_W = 2400.0  # Power scaled by GPU utilization

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
) -> float:
    """
    Computes thermodynamic equilibrium temperature (°C) for a rack based on:
    - Total power dissipation (kW)
    - CRAC cooling effectiveness (0-100%)
    - Adjacent rack heat recirculation
    - Local hot-aisle recirculation factor
    """
    power_kw = total_power_w / 1000.0
    
    # Power temperature rise: ~3.8°C per kW of heat generated
    raw_heat_rise = power_kw * 3.85 * recirculation_factor
    
    # Cooling dampening: higher cooling capacity extracts heat more effectively
    cooling_factor = max(0.20, (cooling_capacity_pct / 100.0))
    effective_heat_rise = raw_heat_rise / (cooling_factor ** 0.85)
    
    # Neighbor thermal coupling: ~15% heat transfer from adjacent racks
    neighbor_diff = max(0.0, neighbor_avg_temp - AMBIENT_TEMP)
    neighbor_influence = 0.14 * neighbor_diff
    
    final_temp = AMBIENT_TEMP + effective_heat_rise + neighbor_influence
    return round(float(np.clip(final_temp, 28.0, 92.0)), 1)


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
        np.random.seed(42)

        # Baseline profiles (CPU %, GPU %, Cooling %, Recirculation Factor)
        base_configs = {
            1:  {"cpu": 45, "gpu": 50, "cooling": 92, "recirc": 1.00, "workload": "BERT Embeddings"},
            2:  {"cpu": 55, "gpu": 60, "cooling": 90, "recirc": 1.02, "workload": "ResNet Vision"},
            3:  {"cpu": 35, "gpu": 40, "cooling": 96, "recirc": 0.95, "workload": "Light Inference"},
            4:  {"cpu": 60, "gpu": 65, "cooling": 88, "recirc": 1.00, "workload": "Audio Transcription"},
            5:  {"cpu": 50, "gpu": 55, "cooling": 90, "recirc": 1.05, "workload": "Recommendation Sys"},
            6:  {"cpu": 65, "gpu": 70, "cooling": 85, "recirc": 1.08, "workload": "Diffusion Model"},
            7:  {"cpu": 25, "gpu": 30, "cooling": 88, "recirc": 1.00, "workload": "Idle / Standby"},
            8:  {"cpu": 50, "gpu": 58, "cooling": 90, "recirc": 1.00, "workload": "NLP Tokenizer"},
            9:  {"cpu": 40, "gpu": 45, "cooling": 94, "recirc": 0.96, "workload": "Vector Search"},
            10: {"cpu": 60, "gpu": 68, "cooling": 86, "recirc": 1.02, "workload": "RL Agent Training"},
            11: {"cpu": 45, "gpu": 52, "cooling": 92, "recirc": 0.98, "workload": "Graph Neural Net"},
            12: {"cpu": 55, "gpu": 62, "cooling": 88, "recirc": 1.00, "workload": "Video Analytics"},
        }

        if scenario_name == "Thermal Hotspot":
            # Scenario: Rack 7 has high GPU availability (70% free, 30% util)
            # but due to hot-aisle air trap & degraded localized cooling, it is already thermally stressed (68.2°C).
            # Rack 3 has 60% free GPU (40% util) with 96% cooling capacity and safe 48.1°C temp.
            base_configs[7] = {"cpu": 25, "gpu": 30, "cooling": 64, "recirc": 1.38, "workload": "Batch Worker"}
            base_configs[6] = {"cpu": 75, "gpu": 82, "cooling": 80, "recirc": 1.15, "workload": "LLM Fine-tuning"}
            base_configs[8] = {"cpu": 70, "gpu": 78, "cooling": 82, "recirc": 1.12, "workload": "Dense Retrieval"}
            base_configs[3] = {"cpu": 35, "gpu": 40, "cooling": 96, "recirc": 0.92, "workload": "API Gateway"}
            base_configs[2] = {"cpu": 40, "gpu": 45, "cooling": 94, "recirc": 0.95, "workload": "Inference Cache"}

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
                s_power = compute_server_power(s_cpu, s_gpu)
                servers.append({
                    "server_id": f"R{r_id}-S{s_idx}",
                    "server_num": s_idx,
                    "cpu_util": round(s_cpu, 1),
                    "gpu_util": round(s_gpu, 1),
                    "gpu_count": GPUS_PER_SERVER,
                    "power_w": s_power,
                    "active_workload": cfg["workload"]
                })
            
            total_rack_power = sum(s["power_w"] for s in servers)
            avg_cpu = float(np.mean([s["cpu_util"] for s in servers]))
            avg_gpu = float(np.mean([s["gpu_util"] for s in servers]))
            
            self.racks[r_id] = {
                "rack_id": r_id,
                "row": RACK_COORDS[r_id][0],
                "col": RACK_COORDS[r_id][1],
                "servers": servers,
                "total_power_w": round(total_rack_power, 1),
                "avg_cpu_util": round(avg_cpu, 1),
                "avg_gpu_util": round(avg_gpu, 1),
                "gpu_available_pct": round(100.0 - avg_gpu, 1),
                "gpu_available_count": int(round((100.0 - avg_gpu) / 100.0 * (SERVERS_PER_RACK * GPUS_PER_SERVER))),
                "cooling_capacity_pct": cfg["cooling"],
                "recirculation_factor": cfg["recirc"],
                "primary_workload": cfg["workload"],
                "temperature": 45.0  # Placeholder, computed next
            }
            temp_estimates[r_id] = 45.0

        # Iterative thermal equilibrium solver (incorporating neighbors)
        for _ in range(4):
            for r_id in range(1, TOTAL_RACKS + 1):
                nbr_ids = get_neighbor_rack_ids(r_id)
                nbr_avg = float(np.mean([temp_estimates[nid] for nid in nbr_ids]))
                t_eq = compute_equilibrium_temperature(
                    self.racks[r_id]["total_power_w"],
                    self.racks[r_id]["cooling_capacity_pct"],
                    nbr_avg,
                    self.racks[r_id]["recirculation_factor"]
                )
                temp_estimates[r_id] = t_eq

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
                "Status": r["status"],
                "Thermal Headroom (°C)": r["thermal_headroom"],
                "GPU Available (%)": r["gpu_available_pct"],
                "GPU Count Free": r["gpu_available_count"],
                "Avg GPU Util (%)": r["avg_gpu_util"],
                "Avg CPU Util (%)": r["avg_cpu_util"],
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

        return {
            "total_racks": TOTAL_RACKS,
            "total_servers": TOTAL_RACKS * SERVERS_PER_RACK,
            "total_gpus": TOTAL_RACKS * SERVERS_PER_RACK * GPUS_PER_SERVER,
            "avg_temp": round(float(np.mean(temps)), 1),
            "max_temp": round(float(np.max(temps)), 1),
            "min_temp": round(float(np.min(temps)), 1),
            "total_power_kw": round(float(np.sum(powers)) / 1000.0, 2),
            "avg_gpu_util": round(float(np.mean(gpu_utils)), 1),
            "hotspot_count": hotspots,
            "warm_count": warm_count,
            "healthy_racks": TOTAL_RACKS - hotspots - warm_count
        }

    def simulate_placement(self, rack_id: int, workload_gpu_pct: float, workload_power_w: float) -> Dict[str, Any]:
        """
        Simulates the physical outcome of placing a workload on a given rack.
        Returns the new state without modifying the persistent simulation.
        """
        curr = self.racks[rack_id]
        new_power_w = curr["total_power_w"] + workload_power_w
        new_gpu_util = min(100.0, curr["avg_gpu_util"] + (workload_gpu_pct / SERVERS_PER_RACK))
        
        # New temperature under modified power
        new_temp = compute_equilibrium_temperature(
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
            "temp_delta": round(new_temp - curr["temperature"], 1),
            "prev_power_w": curr["total_power_w"],
            "new_power_w": round(new_power_w, 1),
            "new_headroom": round(new_headroom, 1),
            "new_status": status,
            "new_status_color": color,
            "new_gpu_util": round(new_gpu_util, 1),
            "is_hotspot": new_temp >= TEMP_CRITICAL,
            "sla_violation": new_temp >= TEMP_CRITICAL
        }


def generate_synthetic_training_data(n_samples: int = 3500) -> pd.DataFrame:
    """
    Generates realistic, physics-grounded synthetic telemetry for ML model training.
    Computes physically consistent current and post-placement temperatures.
    """
    np.random.seed(1337)
    
    # Generate realistic base parameters
    cpu_util = np.random.uniform(15.0, 90.0, n_samples)
    gpu_util = np.random.uniform(10.0, 95.0, n_samples)
    cooling_cap = np.random.uniform(45.0, 98.0, n_samples)
    recirc_factors = np.random.uniform(0.92, 1.35, n_samples)
    
    # Base power derived from CPU/GPU
    base_power = 4 * (400.0 + (cpu_util/100.0)*350.0 + (gpu_util/100.0)*2400.0)
    base_power += np.random.normal(0, 80.0, n_samples)
    base_power = np.clip(base_power, 2000.0, 12500.0)
    
    # Neighbor temperature baseline
    neighbor_temp = 22.0 + (base_power / 1000.0) * 3.5 / ((cooling_cap / 100.0) ** 0.85)
    neighbor_temp += np.random.normal(0, 2.5, n_samples)
    neighbor_temp = np.clip(neighbor_temp, 30.0, 78.0)

    # Current equilibrium temperature based on base power and cooling
    current_temp = []
    for p, c, n_temp, r_fac in zip(base_power, cooling_cap, neighbor_temp, recirc_factors):
        t_curr = compute_equilibrium_temperature(p, c, n_temp, r_fac)
        t_curr += np.random.normal(0, 0.20)
        current_temp.append(round(float(t_curr), 2))
    current_temp = np.array(current_temp)

    # Workload parameters
    workload_gpu_demand = np.random.choice([10.0, 20.0, 35.0, 50.0, 70.0, 90.0], n_samples)
    workload_intensity_val = np.random.choice([1, 2, 3, 4], n_samples)  # 1: Low, 2: Med, 3: High, 4: Very High
    workload_power_est = workload_gpu_demand * 25.0 * (0.8 + 0.15 * workload_intensity_val)
    workload_power_est += np.random.normal(0, 40.0, n_samples)
    workload_power_est = np.clip(workload_power_est, 150.0, 3200.0)
    
    # Target: Equilibrium temperature after placement
    new_total_power = base_power + workload_power_est
    
    target_post_temp = []
    for p, c, n_temp, r_fac in zip(new_total_power, cooling_cap, neighbor_temp, recirc_factors):
        t_post = compute_equilibrium_temperature(p, c, n_temp, r_fac)
        t_post += np.random.normal(0, 0.25)
        target_post_temp.append(round(float(t_post), 2))
        
    df = pd.DataFrame({
        "current_temp": np.round(current_temp, 2),
        "cpu_utilization": np.round(cpu_util, 1),
        "gpu_utilization": np.round(gpu_util, 1),
        "power_watts": np.round(base_power, 1),
        "cooling_capacity": np.round(cooling_cap, 1),
        "neighbor_avg_temp": np.round(neighbor_temp, 2),
        "workload_gpu_demand": np.round(workload_gpu_demand, 1),
        "workload_intensity_val": workload_intensity_val,
        "workload_power_est": np.round(workload_power_est, 1),
        "post_workload_temp": target_post_temp
    })
    
    return df
