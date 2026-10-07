"""
HeatWise: Central Configuration Module
Contains physical constants, thermal thresholds, cooling config, scheduler weights,
and ML config used across the application.
"""

# ==========================================
# SIMULATION CONFIG
# ==========================================
GRID_ROWS = 3
GRID_COLS = 4
TOTAL_RACKS = 12
SERVERS_PER_RACK = 4
GPUS_PER_SERVER = 8

# Power configurations (Watts)
SERVER_IDLE_POWER_W = 400.0
SERVER_MAX_CPU_POWER_W = 350.0
SERVER_MAX_GPU_POWER_W = 2400.0

# Memory configurations
SERVER_MAX_MEMORY_GB = 512.0


# ==========================================
# THERMAL CONFIG
# ==========================================
AMBIENT_TEMP = 22.0
TEMP_NORMAL_MAX = 60.0
TEMP_WARM_MAX = 68.0
TEMP_HIGH_MAX = 74.9
TEMP_CRITICAL = 75.0
TEMP_MAX_SAFE = 80.0

# ==========================================
# ML & BENCHMARK CONFIG
# ==========================================
ML_RANDOM_SEED = 42
SIMULATION_RANDOM_SEED = 42
TELEMETRY_RANDOM_SEED = 1337

# ==========================================
# SCHEDULER CONFIG
# ==========================================
DEFAULT_WEIGHTS = {
    "temp_safety": 0.35,      # Weight for predicted post-placement temperature safety
    "thermal_headroom": 0.25, # Weight for remaining thermal margin below critical threshold
    "gpu_fit": 0.20,          # Weight for computational GPU availability & fit
    "cooling_eff": 0.10,      # Weight for rack cooling capacity
    "neighbor_stability": 0.10 # Weight for thermal stability of surrounding racks
}

# ==========================================
# WORKLOAD CONFIG
# ==========================================
WORKLOAD_PROFILES = {
    "AI Training": {
        "description": "High-intensity multi-epoch model training (PyTorch/JAX)",
        "base_gpu_pct": 50.0,
        "base_mem_pct": 60.0,
        "base_power_w": 1800.0,
        "base_duration_s": 120,
        "intensity_default": "High",
        "badge_color": "#ec4899"
    },
    "AI Inference": {
        "description": "Real-time microservice model inference",
        "base_gpu_pct": 25.0,
        "base_mem_pct": 30.0,
        "base_power_w": 800.0,
        "base_duration_s": 30,
        "intensity_default": "Medium",
        "badge_color": "#06b6d4"
    },
    "LLM Inference": {
        "description": "High memory-bandwidth autoregressive token generation",
        "base_gpu_pct": 40.0,
        "base_mem_pct": 85.0,
        "base_power_w": 1400.0,
        "base_duration_s": 60,
        "intensity_default": "High",
        "badge_color": "#8b5cf6"
    },
    "Image Processing": {
        "description": "Computer vision batch augmentation & filtering",
        "base_gpu_pct": 20.0,
        "base_mem_pct": 45.0,
        "base_power_w": 600.0,
        "base_duration_s": 45,
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
