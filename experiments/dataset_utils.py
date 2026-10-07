"""
HeatWise Experimental Dataset Generation and Splitting Utilities
Supports sequential multi-episode trajectory generation with episode_id and step_idx metadata,
preserving strict state-transition dynamics and anti-leakage guarantees.
"""

import sys
import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import simulation
import ml_model

SCENARIOS = [
    "Normal Operation",
    "GPU Surge",
    "Cooling Degradation",
    "Thermal Hotspot",
    "High AI Workload"
]

METADATA_COLUMNS = ["episode_id", "step_idx", "scenario_name"]

def generate_sequential_dataset(
    n_samples: int = 3500,
    seed: int = config.TELEMETRY_RANDOM_SEED,
    max_steps_per_episode: int = 20
) -> pd.DataFrame:
    """
    Generates a sequential transition dataset from the analytical simulator.
    Maintains trajectory state transitions across episodes while recording
    episode_id, step_idx, and scenario_name.
    """
    rng = np.random.RandomState(seed)
    samples = []
    episode_counter = 0

    while len(samples) < n_samples:
        scenario = SCENARIOS[episode_counter % len(SCENARIOS)]
        sim = simulation.DataCenterSimulation(scenario=scenario)
        episode_counter += 1

        for step in range(1, max_steps_per_episode + 1):
            if len(samples) >= n_samples:
                break

            wl_types = ["AI Training", "AI Inference", "LLM Inference", "Image Processing"]
            wl_type = rng.choice(wl_types, p=[0.35, 0.30, 0.25, 0.10])
            w_int = int(rng.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.30, 0.10]))

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
                    "episode_id": episode_counter,
                    "step_idx": step,
                    "scenario_name": scenario,

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

            # State evolution: neutral randomized placement to explore thermal bounds
            valid_racks = list(range(1, 13))
            winner = rng.choice(valid_racks)
            sim.racks[winner]["total_power_w"] += power_w
            sim.racks[winner]["avg_gpu_util"] = min(100.0, sim.racks[winner]["avg_gpu_util"] + gpu_pct / simulation.SERVERS_PER_RACK)
            sim.racks[winner]["avg_mem_util"] = min(100.0, sim.racks[winner]["avg_mem_util"] + mem_pct / simulation.SERVERS_PER_RACK)
            sim.racks[winner]["gpu_available_pct"] = max(0.0, sim.racks[winner]["gpu_available_pct"] - gpu_pct / simulation.SERVERS_PER_RACK)
            sim.racks[winner]["mem_available_pct"] = max(0.0, sim.racks[winner]["mem_available_pct"] - mem_pct / simulation.SERVERS_PER_RACK)

            for _ in range(4):
                temp_ests = {rid: r["temperature"] for rid, r in sim.racks.items()}
                for rid in range(1, 13):
                    r = sim.racks[rid]
                    nbr = [temp_ests[nid] for nid in r["neighbor_ids"]]
                    inlet, outlet, t_eq = simulation.compute_equilibrium_temperature(
                        r["total_power_w"], r["cooling_capacity_pct"], np.mean(nbr), r["recirculation_factor"]
                    )
                    sim.racks[rid]["temperature"] = t_eq
                    sim.racks[rid]["inlet_temp"] = inlet
                    sim.racks[rid]["neighbor_avg_temp"] = np.mean(nbr)

    df = pd.DataFrame(samples)
    return df


def split_episode_grouped(df: pd.DataFrame, test_size: float = 0.20, seed: int = config.ML_RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset strictly by episode_id. No rows from the same episode
    appear in both train and test.
    """
    episodes = df["episode_id"].unique()
    rng = np.random.RandomState(seed)
    shuffled_episodes = rng.permutation(episodes)

    n_test_episodes = max(1, int(round(len(episodes) * test_size)))
    test_episodes = set(shuffled_episodes[:n_test_episodes])

    train_df = df[~df["episode_id"].isin(test_episodes)].copy()
    test_df = df[df["episode_id"].isin(test_episodes)].copy()

    return train_df, test_df


def split_temporal_blocks(df: pd.DataFrame, train_step_cutoff: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits trajectories temporally within episodes:
    Train = step_idx <= train_step_cutoff (early trajectory phase)
    Test  = step_idx > train_step_cutoff (sustained/elevated trajectory phase)
    """
    train_df = df[df["step_idx"] <= train_step_cutoff].copy()
    test_df = df[df["step_idx"] > train_step_cutoff].copy()
    return train_df, test_df
