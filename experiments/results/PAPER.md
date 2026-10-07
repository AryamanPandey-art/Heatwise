# HeatWise: Thermal-Aware Workload Placement for High-Density AI Data Centers Using a Reduced-Order Surrogate Model

**Aryaman Pandey**  
*Google DeepMind / HeatWise Research Team*  
**Artifact & Software Release:** `Commit 76fe215daee5462b55515045a2495c2725179fe6`  
**Repository:** [github.com/Yurirain/Heatwise](https://github.com/Yurirain/Heatwise)

---

## Abstract

Modern artificial intelligence (AI) data centers face severe thermal challenges driven by extreme power densities and localized recirculation. Conventional workload schedulers—such as Round Robin and Least Loaded—prioritize compute availability while remaining thermally blind, frequently scheduling power-intensive accelerators into localized thermal pockets and exacerbating hot-aisle recirculation. We present **HeatWise**, a simulation-based, thermal-aware workload placement framework designed for high-density accelerator facilities. HeatWise integrates an analytical lumped-parameter reduced-order thermal model with an empirical Random Forest surrogate model to predict post-placement rack equilibrium temperatures prior to physical task dispatch. A constrained multi-factor placement heuristic then balances predicted thermal safety (35%), thermal headroom (25%), compute capacity fit (20%), local heat removal capacity (10%), and neighbor thermal stability (10%).

Across closed-loop dynamic trajectory evaluations against an analytical physical oracle over 250 scheduling decisions, the production surrogate model achieves a dynamic Mean Absolute Error (MAE) of **1.867°C**, a Top-1 placement agreement of **79.60%**, a Top-3 overlap of **100.0%**, and a mean optimality gap of **+0.473°C**, with zero unsafe placements into candidate racks exceeding the 75.0°C hotspot threshold. In controlled multi-scheduler benchmark simulations under an invariant, decoupled workload stream, HeatWise reduces spatial cluster temperature variance to **8.9°C standard deviation** (a **15.2% variance reduction** relative to Round Robin at 10.5°C), while maintaining zero placement failures. Furthermore, the serialized production deployment (`n_jobs=1`) reduces single-rack inference latency to **2.15 ms** and artifact initialization to **23.64 ms**, demonstrating practical viability for real-time cluster admission control.

---

## 1. Introduction

The rapid scaling of deep neural network training and large language model (LLM) serving has driven data center power densities beyond 40–100 kW per rack. Modern high-density compute nodes dissipate massive heat fluxes across compact form factors, creating steep localized thermal gradients and non-uniform hot-aisle recirculation patterns.

Traditional cluster orchestrators (e.g., Kubernetes, SLURM) schedule jobs primarily based on static compute metrics: accelerator availability, memory headroom, and CPU utilization. This thermal blindness leads to severe operational inefficiencies:
1. **Accelerated Hotspot Formation:** Scheduling a burst of compute-heavy workloads onto a rack with degraded heat dissipation or elevated ambient recirculation triggers localized thermal hotspots exceeding critical thresholds (75.0°C).
2. **Thermal SLA Violations:** Operating temperatures climbing toward or beyond maximum safe limits (80.0°C) force accelerator thermal throttling, degrading application throughput and threatening hardware longevity.
3. **Spatial Thermal Variance:** Uneven heat distribution creates wide temperature differentials across the facility floor, forcing cooling infrastructure to over-provision airflow to satisfy the hottest node.

Full-physics Computational Fluid Dynamics (CFD) simulations accurately capture Navier-Stokes thermal transport but require minutes to hours per iteration, rendering them unusable for online cluster scheduling where admission decisions must resolve within milliseconds.

HeatWise addresses this gap through a surrogate modeling approach:
- We formulate a **lumped-parameter reduced-order thermal model** capturing conductive, convective, and cross-rack recirculation physics across a 12-rack facility.
- We train an offline **Random Forest Regressor** surrogate on 10,000 synthetic sequential transition vectors using an exact **17-feature leakage-free pre-placement schema**.
- We develop a **multi-factor placement engine** that scores racks across thermal safety, headroom, compute fit, cooling capacity, and neighbor stability, enforcing strict safety thresholds ($T_{crit} = 75.0^\circ\text{C}$, $T_{max\_safe} = 80.0^\circ\text{C}$).

```
+-----------------------------------------------------------------------------------+
|                            HEATWISE PIPELINE OVERVIEW                             |
+-----------------------------------------------------------------------------------+
| [Incoming Job] -> [Pre-Placement Feature Extraction] (17 Leakage-Free Variables)   |
|                         |                                                         |
|                         v                                                         |
| [Surrogate ML Model] -> Predicts Post-Placement T_eq for Racks 1..12 (2.15 ms)    |
|                         |                                                         |
|                         v                                                         |
| [Heuristic Engine]   -> Filters Unsafe Racks (T_pred >= 75°C) & Infeasible Compute|
|                         Composite Score: 35% Safe, 25% Head, 20% Fit, 10% Cool/Nbr|
|                         |                                                         |
|                         v                                                         |
| [Target Allocation]  -> Dispatches Job & Authoritative Simulator Advances Physics |
+-----------------------------------------------------------------------------------+
```

---

## 2. Related Work

Prior research on thermal-aware scheduling broadly partitions into three methodologies:

1. **Reactive Thermal Throttling:** Hardware-level dynamic voltage and frequency scaling (DVFS) reduces clock speeds when internal junctions reach thermal limits. While reactive DVFS protects silicon, it causes unpredictable job latency spikes and fails to resolve the underlying hot-aisle congestion.
2. **Online CFD and Computational Geometry:** Numerical approximations of airflow dynamics provide high fidelity but remain computationally prohibitive for cluster-scale dynamic scheduling ($>10^2$ seconds per step).
3. **Data-Driven & Reduced-Order Modeling:** Machine learning surrogates (neural networks, regression trees) have emerged as viable approximations of thermal behavior. However, prior models often suffer from **temporal feature leakage** (conditioning on post-placement telemetry unavailable during admission) or lack **closed-loop dynamic validation** in sequential state-machine simulations.

HeatWise advances reduced-order scheduling by combining strict anti-leakage feature design, closed-loop physical oracle validation, and millisecond-scale execution in an open-source, reproducible framework.

---

## 3. System Architecture

The simulated data center represents an enterprise AI accelerator facility organized as a 3x4 physical grid of 12 high-density racks.

```
+-----------------------------------------------------------------------+
|                 DATA CENTER TOPOLOGY (3x4 PHYSICAL GRID)              |
+-----------------------------------------------------------------------+
|  [Rack 1] (0,0)  |  [Rack 2] (0,1)  |  [Rack 3] (0,2)  |  [Rack 4] (0,3)  |
|  ----------------+------------------+------------------+----------------  |
|  [Rack 5] (1,0)  |  [Rack 6] (1,1)  |  [Rack 7] (1,2)  |  [Rack 8] (1,3)  |
|  ----------------+------------------+------------------+----------------  |
|  [Rack 9] (2,0)  |  [Rack 10] (2,1) |  [Rack 11] (2,2) |  [Rack 12] (2,3) |
+-----------------------------------------------------------------------+
```

### Facility Specifications
- **Grid Dimensions:** 3 rows $\times$ 4 columns = 12 racks ($TOTAL\_RACKS = 12$).
- **Compute Density:** 4 servers per rack ($SERVERS\_PER\_RACK = 4$); 8 GPUs per server ($GPUS\_PER\_SERVER = 8$).
- **Total Silicon:** 32 GPUs per rack; **384 GPUs cluster-wide**.
- **Memory Density:** 512 GB memory per server; 2,048 GB memory per rack; **24.5 TB cluster-wide**.
- **Power Envelope:**
  - Server Idle Power: $P_{idle} = 400.0\text{ W}$ ($1.6\text{ kW}$ baseline per rack).
  - Server Peak CPU: $P_{cpu,max} = 350.0\text{ W}$.
  - Server Peak GPU: $P_{gpu,max} = 2,400.0\text{ W}$ ($300\text{ W}$ per accelerator).
  - Maximum Rack Power: $P_{rack,max} = 12.6\text{ kW}$ ($151.2\text{ kW}$ cluster capacity).

---

## 4. Thermal Model

The ground-truth thermal physics are governed by a reduced-order lumped-parameter model implemented in [`simulation.py`](file:///Users/Yurirain/Downloads/Heatwise/simulation.py).

### 4.1 Server Power Computation
Server power consumption $P_{srv}$ couples CPU utilization $u_{cpu} \in [0, 100]$ and GPU utilization $u_{gpu} \in [0, 100]$ via non-linear quadratic scaling:

$$P_{srv} = P_{idle} + \left(\frac{u_{cpu}}{100}\right) P_{cpu,max} + \left[0.15\left(\frac{u_{gpu}}{100}\right) + 0.85\left(\frac{u_{gpu}}{100}\right)^2\right] P_{gpu,max}$$

Rack power $P_{rack}$ is the sum of its four constituent servers:

$$P_{rack} = \sum_{s=1}^{4} P_{srv,s}$$

### 4.2 Equilibrium Temperature Formulation
For rack $i$ with total power $P_i$ (Watts), cooling capacity percentage $C_i \in [10, 100]$, average neighbor temperature $\bar{T}_{nbr,i}$ (°C), and recirculation factor $R_i \in [0.8, 1.5]$:

1. **Inlet Air Temperature ($T_{inlet}$):** Air arriving at rack $i$ mixes ambient CRAC supply ($T_{amb} = 22.0^\circ\text{C}$) with recirculated exhaust from adjacent physical neighbors:

$$T_{inlet,i} = T_{amb} + 0.35 \times (\bar{T}_{nbr,i} - T_{amb}) \times (R_i - 1.0)$$

2. **Heat Rise ($\Delta T$):** The steady-state thermal rise across the chassis depends on dissipated power mitigated by effective convective airflow:

$$\Delta T_i = \frac{P_i}{1000.0} \times \frac{4.2}{\left(\frac{C_i}{100.0}\right)^{0.85}} \times R_i$$

3. **Chassis Outlet Temperature ($T_{outlet}$):**

$$T_{outlet,i} = T_{inlet,i} + \Delta T_i$$

4. **Equilibrium Temperature ($T_{eq}$):** Equilibrium rack temperature reflects internal junction-to-ambient thermal resistance:

$$T_{eq,i} = T_{inlet,i} + 0.65 \times \Delta T_i$$

### 4.3 Operational Safety Thresholds
- **$T_{crit} = 75.0^\circ\text{C}$ (`config.TEMP_CRITICAL`):** Critical hotspot threshold. Any rack exceeding 75.0°C is classified as an active hotspot.
- **$T_{max\_safe} = 80.0^\circ\text{C}$ (`config.TEMP_MAX_SAFE`):** Maximum safe operating ceiling. Rack temperatures at or above 80.0°C constitute a thermal SLA violation.

---

## 5. ML Dataset and Feature Engineering

### 5.1 Strict Anti-Leakage Feature Schema
The surrogate predictor uses **17 pre-placement features** Observable strictly *before* placement occurs. Post-placement equilibrium telemetry, actual outlet temperatures, and post-workload power draws are strictly excluded from the feature matrix.

| # | Feature Name | Physical Interpretation | Units / Range |
| :-: | :--- | :--- | :-: |
| 1 | `cpu_utilization` | Current rack average CPU load | % (0–100) |
| 2 | `gpu_utilization` | Current rack average GPU load | % (0–100) |
| 3 | `memory_utilization` | Current rack average RAM load | % (0–100) |
| 4 | `inlet_temperature` | Measured server inlet air temperature | °C (20–40) |
| 5 | `local_power_kw` | Current instantaneous rack power draw | kW (1.6–12.6) |
| 6 | `neighbor_temperature` | Mean temperature of physical grid neighbors | °C (30–80) |
| 7 | `neighbor_utilization` | Mean GPU utilization of physical grid neighbors | % (0–100) |
| 8 | `cooling_capacity` | Effective local CRAC airflow capacity | % (10–100) |
| 9 | `recirculation_factor` | Local geometric airflow recirculation index | Unitless (0.8–1.5) |
| 10 | `workload_AI_Training` | One-hot indicator: Job is distributed training | Binary {0, 1} |
| 11 | `workload_AI_Inference`| One-hot indicator: Job is batch inference | Binary {0, 1} |
| 12 | `workload_LLM` | One-hot indicator: Job is LLM generation | Binary {0, 1} |
| 13 | `workload_Image_Processing` | One-hot indicator: Job is vision preprocessing | Binary {0, 1} |
| 14 | `workload_intensity_val`| Intensity classification multiplier | Integer (1–4) |
| 15 | `workload_gpu_demand` | Accelerator demand of incoming task | % (10–100) |
| 16 | `workload_mem_demand` | Memory demand of incoming task | % (10–100) |
| 17 | `workload_power_est` | Analytical added power estimate | Watts (300–3,200) |
| — | **`post_workload_temp`** | **Target variable: Post-placement $T_{eq}$** | **°C (Continuous)** |

### 5.2 Synthetic Sequential Trajectory Generation
To ensure operational validity, training telemetry is not drawn from uniform random noise. Instead, [`simulation.py:generate_synthetic_training_data`](file:///Users/Yurirain/Downloads/Heatwise/simulation.py#L376) executes multi-step sequential state-machine trajectories across five realistic operational scenarios:
1. **Normal Operation:** Balanced compute distribution across the 12 racks ($C_i = 92\%$, $R_i = 0.98$).
2. **GPU Surge:** Sudden compute burst across bottom-row racks 9–12 ($u_{gpu} = 88\%$).
3. **Cooling Degradation:** Middle-row CRAC airflow reduction on racks 5–8 ($C_i \in [42\%, 48\%]$, $R_i = 1.25$).
4. **Thermal Hotspot:** Trapped air recirculation at rack 7 ($C_7 = 64\%$, $R_7 = 1.38$, initial $T_7 = 68.2^\circ\text{C}$).
5. **High AI Workload:** Top-left cluster saturation across racks 1, 2, 5, 6, 10.

---

## 6. HeatWise Placement Algorithm

When a workload arrives, the placement engine evaluates all 12 racks using candidate filtering and a constrained multi-factor objective.

### 6.1 Feasibility and Candidate Filtering
For each candidate rack $i \in \{1, \dots, 12\}$:
1. **Hardware Compute Feasibility:**
   $$GPU_{avail,i} \ge \frac{GPU_{demand}}{4} \quad \text{and} \quad MEM_{avail,i} \ge \frac{MEM_{demand}}{4}$$
2. **Surrogate Thermal Admission:**
   The surrogate model predicts $T_{pred,i} = f_{ML}(\mathbf{x}_i, \mathbf{w})$. Candidate $i$ is admitted to the primary candidate pool $\mathcal{P}_{safe}$ strictly if:
   $$T_{pred,i} < T_{crit} \quad (75.0^\circ\text{C})$$

### 6.2 Multi-Factor Scoring Formulation
If $\mathcal{P}_{safe} \neq \emptyset$, candidates in $\mathcal{P}_{safe}$ are evaluated by a weighted linear combination of five normalized sub-scores ($S \in [0, 100]$):

$$Score_i = w_{safe} S_{safe,i} + w_{head} S_{head,i} + w_{fit} S_{fit,i} + w_{cool} S_{cool,i} + w_{nbr} S_{nbr,i}$$

Scoring weights are calibrated to prioritize thermal reliability while maintaining hardware utilization:
- **Thermal Safety ($w_{safe} = 0.35$):**
  $$S_{safe,i} = \text{clip}\left(\frac{75.0 - T_{pred,i}}{75.0 - 35.0} \times 100.0, \; 0, \; 100\right)$$
- **Thermal Headroom ($w_{head} = 0.25$):**
  $$S_{head,i} = \text{clip}\left(\frac{75.0 - T_{pred,i}}{75.0 - 22.0} \times 100.0, \; 0, \; 100\right)$$
- **Compute Resource Fit ($w_{fit} = 0.20$):**
  $$S_{fit,i} = GPU_{avail,i}$$
- **Cooling Capacity Efficiency ($w_{cool} = 0.10$):**
  $$S_{cool,i} = C_i$$
- **Neighbor Thermal Stability ($w_{nbr} = 0.10$):**
  $$S_{nbr,i} = \text{clip}\left(\frac{75.0 - \bar{T}_{nbr,i}}{75.0 - 35.0} \times 100.0, \; 0, \; 100\right)$$

### 6.3 Additive Fallback Penalty
If continuous heavy load pushes all feasible racks above 75.0°C ($\mathcal{P}_{safe} = \emptyset$), HeatWise activates a fallback penalty to select the least dangerous feasible candidate:

$$Penalty_i = 45.0 + 5.0 \times (T_{pred,i} - 75.0)$$
$$Score_{fallback,i} = Score_i - Penalty_i$$

---

## 7. Experimental Methodology

To evaluate HeatWise rigorously, we implement an isolated research framework ([`experiments/`](file:///Users/Yurirain/Downloads/Heatwise/experiments/)) completely decoupled from the production runtime.

### 7.1 Analytical Oracle Definition
The analytical oracle is the closed-loop physical thermal simulator. For any candidate rack $i$, the oracle computes exact equilibrium temperature $T_{oracle,i}$ by executing `simulate_placement()` through the thermal equations.

### 7.2 Decision Quality Metrics
Over a 250-decision dynamic sequence:
- **Dynamic MAE:** Mean absolute deviation between surrogate prediction and physical oracle.
- **Top-1 Agreement:** Percentage of decisions where the surrogate's top-ranked rack exactly matches the oracle's top-ranked rack.
- **Top-3 Overlap:** Percentage of steps where the surrogate's top selection is included within the oracle's top-3 set.
- **Spearman Rank Correlation ($\rho$):** Monotonic ranking agreement across all 12 candidate racks.
- **Mean Optimality Gap:** Thermal penalty incurred relative to the physical oracle's optimal choice:
  $$\Delta T_{opt} = T_{oracle}(rack_{ML}) - T_{oracle}(rack_{oracle})$$
- **Materially Worse Rate ($>2.0^\circ\text{C}$):** Fraction of steps where $\Delta T_{opt} > 2.0^\circ\text{C}$.
- **Unsafe Placements:** Allocations dispatched to racks where actual $T_{eq} \ge 75.0^\circ\text{C}$.
- **Hardware Failures:** Allocations violating memory or GPU capacity.

---

## 8. Static ML Validation

We evaluate surrogate architectures on offline sequential telemetry holdout splits.

### 8.1 Model A: Grouped Holdout Research Model
To eliminate temporal autocorrelation leakage, Model A partitions 10,000 transition vectors into **8,080 training samples** and **1,920 test samples** grouped strictly by complete scenario episodes.

| Architecture | Trees | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Dynamic MAE | Dynamic Top-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Model A)** | 120 | **0.460°C** | **0.830°C** | **99.51%** | **1.931°C** | **81.20%** |
| ExtraTreesRegressor | 120 | 0.443°C | 0.812°C | 99.53% | 2.138°C | 78.40% |
| HistGradientBoosting | 100 | 0.493°C | 0.892°C | 99.43% | 2.502°C | 72.80% |

### 8.2 Feature Importance Distribution
Tree variance analysis demonstrates that thermal predictions are driven primarily by power and cooling variables:
1. `Rack Power (local_power_kw)`: **65.76%**
2. `Cooling Capacity (cooling_capacity)`: **18.03%**
3. `Recirculation Factor (recirculation_factor)`: **6.30%**
4. `Estimated Added Power (workload_power_est)`: **3.51%**
5. `Workload GPU Demand (workload_gpu_demand)`: **2.66%**
6. Remaining 12 features combined: **3.74%**

---

## 9. Dynamic Closed-Loop Validation

### 9.1 Model A Multi-Seed Robustness (Research Artifact)
Model A was evaluated across five random seeds (Seeds 42, 101, 202, 303, 404) to establish dynamic placement stability:

$$\text{Model A Top-1 Agreement} = 81.20 \pm 2.28\%$$
$$\text{Model A Dynamic MAE} = 1.931 \pm 0.016^\circ\text{C}$$
$$\text{Model A Spearman } \rho = 0.548 \pm 0.038$$
$$\text{Model A Optimality Gap} = +0.473 \pm 0.170^\circ\text{C}$$

### 9.2 Model B Dynamic Placement Evaluation (Final Production Artifact)
Model B is trained on the full 10,000-sample trajectory dataset for deployment. Evaluated on the standardized 250-decision closed-loop benchmark (Seed 101), Model B delivers strong decision quality:

| Evaluation Metric | Baseline RF (3,500 Samples) | Candidate Model B (10,000 Samples) | Improvement / Status |
| :--- | :---: | :---: | :---: |
| **Dynamic MAE** | 3.449°C | **1.867°C** | **-45.9% Error Reduction** |
| **Top-1 Agreement** | 40.0% | **79.60%** | **+39.6% Placement Match** |
| **Top-3 Overlap** | 80.0% | **100.0%** | **Zero Top-3 Misses** |
| **Spearman Rank Correlation ($\rho$)** | 0.217 | **0.580** | **+167.3% Ranking Concordance** |
| **Mean Optimality Gap** | +3.580°C | **+0.473°C** | **-86.8% Thermal Penalty** |
| **Materially Worse Rate ($>2.0^\circ\text{C}$)** | 60.0% | **9.20%** | **-84.7% Outlier Reduction** |
| **Catastrophic Rate ($>5.0^\circ\text{C}$)** | 10.0% | **0.00%** | **Zero Catastrophic Errors** |
| **Unsafe Placements ($\ge 75.0^\circ\text{C}$)** | 0 | **0** | **100% Thermal Adherence** |
| **Hardware Infeasible Placements** | 0 | **0** | **100% Resource Feasible** |

*(Note: Model A's 5-seed metric of 81.20% is specific to the 8k-grouped holdout model and is distinct from Model B's validated single-run deployment result of 79.60%.)*

---

## 10. Scheduler Benchmark

### 10.1 Primary Canonical Benchmark: Normalized Isolated Stream (`seed=100`)
The canonical benchmark executes a sequential stream of 15 incoming AI workloads (total added IT demand: 16,690.0 W) across three schedulers starting from the `"Thermal Hotspot"` scenario.

```
+----------------------------------------------------------------------------------------------------+
|                                 CANONICAL BENCHMARK COMPARISON                                     |
+----------------------+--------------------+--------------------+-----------------------------------+
| Metric               | Round Robin        | Least Loaded       | HeatWise 10k (Production Model B) |
+----------------------+--------------------+--------------------+-----------------------------------+
| Peak Equilibrium Temp| 84.1°C             | 84.1°C             | 84.1°C (No peak temp regression)  |
| Cluster Temp Std Dev | 10.5°C             | 9.3°C              | 8.9°C (-15.2% variance vs. RR)    |
| Average Cluster Temp | 64.8°C             | 64.8°C             | 64.6°C (Coolest overall)          |
| Average Headroom     | 11.8°C             | 11.4°C             | 11.6°C (+0.2°C vs. Least Loaded)  |
| Hotspots (>=75.0°C)  | 3                  | 2                  | 2 (33.3% fewer than Round Robin)  |
| SLA Violations (>=80)| 5                  | 4                  | 4 (20.0% fewer than Round Robin)  |
| Placement Failures   | 0                  | 0                  | 0 (0 placement failures)          |
| Total Cluster Power  | 96.97 kW           | 96.97 kW           | 96.97 kW                          |
| Total Cluster Energy | 363.63 kWh         | 363.63 kWh         | 363.63 kWh                        |
+----------------------+--------------------+--------------------+-----------------------------------+
```

### 10.2 Thermal Performance Discussion
- **Thermal Variance Reduction:** HeatWise achieves a spatial standard deviation of **8.9°C** across the 12 racks, compared to 10.5°C for Round Robin and 9.3°C for Least Loaded. This reflects balanced load distribution away from trapped air pockets.
- **Hotspot and SLA Mitigation:** Under heavy burst arrivals, HeatWise records **2 hotspots** and **4 SLA violations**, matching Least Loaded and outperforming Round Robin (3 hotspots, 5 SLA violations).
- **Peak Temperature:** All three schedulers record a peak rack temperature of 84.1°C at Rack 6 due to sustained workload density. **84.1°C is not thermally safe**, exceeding the maximum safe ceiling of 80.0°C. HeatWise demonstrates that under extreme localized cooling constraints, intelligent placement mitigates variance and violation counts but cannot overcome absolute cooling capacity exhaustion.
- **Power & Energy Invariance:** Total IT cluster power (96.97 kW) and energy (363.63 kWh) are identical across all schedulers because all 15 workloads were successfully placed without dropped jobs. **HeatWise does not reduce computing energy; it redistributes dissipated thermal power.**

### 10.3 Historical Stream A Compatibility
To verify that Model B introduced no behavioral regression relative to earlier project milestones, Model B was evaluated against the legacy contaminated workload stream (25,180.0 W added power with 3 "Very High" AI Training workloads):
- **Round Robin:** 84.1°C peak, 9.3°C SD, 3 hotspots, 5 SLA violations, 1 failure, 104.1 kW power, 390.37 kWh energy.
- **Least Loaded:** 92.0°C peak, 11.3°C SD, 3 hotspots, 6 SLA violations, 0 failures, 104.1 kW power, 390.37 kWh energy.
- **HeatWise (Legacy 3.5k):** 84.1°C peak, 7.5°C SD, 2 hotspots, 4 SLA violations, 0 failures, 104.1 kW power, 390.37 kWh energy.
- **HeatWise (Migrated 10k):** **84.1°C peak, 7.5°C SD, 2 hotspots, 4 SLA violations, 0 failures, 104.1 kW power, 390.37 kWh energy.**
- **Decision Concordance:** 14 out of 15 identical rack placements (**93.3% Top-1 match**).

This confirms that the numerical difference between 104.1 kW and 96.97 kW was driven entirely by workload stream variance rather than model regression.

---

## 11. Runtime & Production Validation

The production software implementation was audited and released in commit `76fe215daee5462b55515045a2495c2725179fe6`.

### 11.1 Latency Optimization (`n_jobs=1`)
Profiling revealed that scikit-learn's multiprocessing thread pool (`n_jobs=-1`) incurs substantial IPC synchronization overhead on single-vector placement queries:

| Execution Stage | Multiprocessing (`n_jobs=-1`) | Serialized (`n_jobs=1`) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| Single Rack Placement Query | 11.20 ms | **2.15 ms** | **5.2× Speedup** |
| Full 12-Rack Evaluation Pass | 135.00 ms | **26.56 ms** | **5.1× Speedup** |
| Cold Model Artifact Loading | 374.94 ms (In-Memory Fit) | **23.64 ms** (`joblib.load`) | **15.9× Speedup** |

### 11.2 Software Robustness
- **Unit Testing:** 15 out of 15 unit tests pass in **0.556s** ([`tests/`](file:///Users/Yurirain/Downloads/Heatwise/tests/)).
- **UI Verification:** All 6 Streamlit application tabs pass programmatic end-to-end testing with **0 exceptions** (`streamlit.testing.v1.AppTest`).
- **Module Hygiene:** Runtime memory inspection confirms **zero imports** from `experiments/` during production scheduling.
- **Failure Semantics:** Corrupted artifacts raise loud `RuntimeError` exceptions; legacy in-memory training fallback is preserved strictly when the artifact file is absent.

---

## 12. Limitations

1. **Synthetic Telemetry:** Data is generated by a lumped-parameter physical simulator rather than physical sensor telemetry from a live production facility.
2. **Reduced-Order Thermal Dynamics:** The simulator uses steady-state equilibrium approximations with lumped convective mixing. It does not resolve transient 3D Navier-Stokes turbulence, humidity variations, or fan speed hysteresis curves.
3. **Discrete Facility Scale:** The evaluation is scoped to a 12-rack, 384-GPU facility; multi-room hyperscale data centers require hierarchical surrogate partitioning.
4. **Thermal Exhaustion Limits:** When continuous cluster utilization exceeds cooling capacity, HeatWise cannot prevent SLA violations without job rejection or active workload deferral.

---

## 13. Conclusion

HeatWise demonstrates that simulation-grounded machine learning surrogates enable practical, millisecond-scale thermal intelligence in AI cluster scheduling. By predicting post-placement equilibrium temperatures using 17 leakage-free features, HeatWise achieves a **79.60% dynamic Top-1 agreement** with an analytical physical oracle, reduces spatial thermal variance to **8.9°C standard deviation** (a **15.2% reduction** relative to Round Robin), and eliminates allocation failures without incurring runtime CFD latency. The finalized production artifact delivers **2.15 ms per-rack inference** while maintaining strict software isolation and physical determinism.

---

## 14. Appendix: Reproducibility Notes

### Benchmark RNG Decoupling Discovery
During staging release audits, an RNG coupling anomaly was identified in legacy benchmark code:
```python
# Legacy run_batch_simulation_benchmark:
np.random.seed(seed)                 # Seed set to 100
predictor = get_trained_predictor()  # Legacy fit called np.random.seed(42) and churned 3.5k samples!
workload_stream = [...]              # Sampled from contaminated post-training RNG state
```
In the released codebase ([`heatwise.py:314-326`](file:///Users/Yurirain/Downloads/Heatwise/heatwise.py#L314-L326)), benchmark workload generation is permanently isolated using a dedicated generator:
```python
predictor = get_trained_predictor()
rng = np.random.RandomState(seed)    # Dedicated, decoupled RNG stream
workload_stream = [rng.choice(...) for _ in range(num_workloads)]
```
This guarantees the mathematical invariant:
$$\text{Workload Stream} = f(\text{benchmark\_seed}) \quad \text{independent of model lifecycle}$$

### Artifact Verification
The production model artifact can be independently verified via SHA-256:
```bash
$ shasum -a 256 models/heatwise_rf_10k.joblib
5e35ab92623a5cb8c1e640b5bbd78adaa52c149f68c200cc5f13e1f87787c1de
```
Execution of unit tests and benchmark:
```bash
python3 -m unittest discover tests
python3 run_bench.py
```
