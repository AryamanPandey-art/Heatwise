# HeatWise: Intelligent Thermal-Aware Workload Placement in AI Data Centers

**HeatWise** is a simulation-based, thermal-aware AI workload placement prototype designed for high-density AI data centers. Traditional workload schedulers (such as Least-Loaded or Round Robin) primarily consider computational availability (CPU/GPU load and memory) and are blind to localized thermal congestion and cooling constraints. HeatWise uses machine learning to predict post-placement equilibrium temperatures, evaluate thermal headroom and neighboring rack heat recirculation, and make transparent, explainable placement decisions.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ (tested up to Python 3.13)
- Pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Heatwise/
├── app.py                 # Streamlit UI dashboard, KPI cards, visual pipelines, tabs
├── simulation.py          # 12-rack x 4-server AI data center thermal physics simulation
├── ml_model.py            # Random Forest ML model predicting post-workload temperature
├── heatwise.py            # HeatWise Decision Engine & Baseline Schedulers (RR, Least-Loaded)
├── visualization.py       # Plotly 2D thermal heatmaps, before/after comparisons, radar plots
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation, demo script & presentation guide
```

---

## 🧩 Module Breakdown

### 1. `simulation.py` (Physical Simulation)
- Simulates a 12-rack high-density AI data center organized in a 3x4 grid (48 GPU servers, 384 GPUs total).
- Implements physical equations correlating CPU/GPU utilization, dynamic power draw (400W–2800W per server), CRAC cooling capacity (0–100%), and 2D spatial neighboring rack heat recirculation.
- Contains 5 preset scenarios: *Normal Operation*, *High AI Workload*, *Thermal Hotspot (Demo Scenario)*, *Cooling Degradation*, and *GPU Surge*.

### 2. `ml_model.py` (Thermal ML Predictor)
- Implements a `RandomForestRegressor` trained on 3,500 physics-grounded synthetic telemetry points.
- Features: current temperature, CPU%, GPU%, power (W), cooling capacity (%), neighbor average temperature, and workload demand.
- Target: Post-placement equilibrium temperature (°C).
- Generates accuracy metrics (MAE ~0.4°C, RMSE ~0.6°C, $R^2 > 98\%$) and feature importance distributions.

### 3. `heatwise.py` (Decision Engine & Schedulers)
- Defines AI Workload profiles (*AI Training*, *AI Inference*, *LLM Inference*, *Image Processing*) across intensities and priority levels.
- Implements conventional baseline schedulers: **Round Robin** and **Least-Loaded**.
- Implements **HeatWise Decision Engine**: filters candidates, applies ML thermal predictions, enforces safety penalties ($T \ge 75^\circ\text{C}$), and calculates a composite score:
  $$\text{Score} = w_{\text{temp}} S_{\text{temp}} + w_{\text{head}} S_{\text{head}} + w_{\text{gpu}} S_{\text{gpu}} + w_{\text{cool}} S_{\text{cool}} + w_{\text{nbr}} S_{\text{nbr}} - \text{Penalty}$$
- Provides full human-readable explainability and contrast reasoning.
- Includes a batch simulation engine evaluating multi-job streams.

### 4. `visualization.py` (Interactive Plotly Visualizer)
- 2D Data Center Heatmap (3x4 grid) with temperature color grading and status badges.
- Side-by-side **Before Placement vs. After Placement** thermal comparison heatmaps.
- Multi-criteria Radar and Ranked Score Bar charts.
- Algorithm benchmark comparison bar charts and trajectory line charts.
- ML scatter plot ($y=x$ fit) and feature importance bar chart.

### 5. `app.py` (Streamlit Dashboard)
- Polished dark cyberpunk-datacenter theme with custom CSS and glassmorphic cards.
- 5 intuitive navigation tabs:
  1. **🖥️ Live Operations & Placement**
  2. **📊 Algorithm Comparison & Benchmarks**
  3. **🧠 ML Thermal Prediction Model**
  4. **🏛️ System Architecture**
  5. **ℹ️ About HeatWise & Presentation Script**

---

## 🎯 Key Demonstration Scenario: Rack 7 vs Rack 3

In the **Thermal Hotspot** scenario:
- **Rack 7**: Has **70% free GPU availability** (only 30% utilized), but due to local airflow recirculation and degraded cooling, its current temperature is **68.2°C**.
- **Rack 3**: Has **60% free GPU availability** (40% utilized) with **96% cooling capacity** and a safe temperature of **48.1°C**.
- **Least-Loaded Algorithm Result**: Blindly selects **Rack 7** because it has higher free GPU capacity. Placing a high-intensity AI training job surges Rack 7's temperature to **78.4°C (CRITICAL HOTSPOT)**!
- **HeatWise Result**: Predicts the thermal risk, rejects Rack 7, and places the workload on **Rack 3**, maintaining a safe post-placement temperature of **56.1°C**.

---

## 🎙️ 2-Minute Presentation Walkthrough Script

1. **Introduction (15s)**:
   > *"Good morning Professor. Today I am presenting HeatWise, an intelligent thermal-aware workload placement system for high-density AI data centers. Traditional schedulers prioritize compute availability and are blind to heat, which leads to hotspot formation and throttling. HeatWise uses ML to predict thermal impact before placing workloads."*
2. **Show Telemetry (20s)**:
   > *"Here is our 12-rack virtual data center running 48 GPU servers. Notice on the 2D Heatmap that different racks experience different localized temperatures and cooling efficiencies."*
3. **Show the Conflict Scenario (35s)**:
   > *"Notice Rack 7: it has 70% free GPU capacity, but is already warm at 68°C. Rack 3 has 60% free GPU but is cool at 48°C. A standard Least-Loaded scheduler chooses Rack 7. Let's click 'Find Best Placement'."*
4. **Explain HeatWise Decision (30s)**:
   > *"Our Random Forest model predicts that placing the workload on Rack 7 would spike temperature to 78°C—causing a thermal hotspot and SLA violation! HeatWise penalizes this risk and selects Rack 3 instead, keeping the cluster at a safe 56°C. Notice the Before vs. After Heatmap and the explainability breakdown."*
5. **Show Multi-Job Benchmark (20s)**:
   > *"On Tab 2, we simulate a stream of 15 incoming AI workloads. Least-Loaded and Round Robin suffer repeated hotspots and SLA violations, whereas HeatWise completely eliminates thermal violations and balances cluster heat."*

---

## 🔬 Academic Positioning & Scope

- **Novelty Disclaimer**: Thermal-aware scheduling and ML temperature prediction exist in IEEE/ACM data center literature. HeatWise contributes an **explainable, rack-level simulation prototype** that integrates predicted post-placement thermal impact, neighboring rack recirculation, power, cooling capacity, and thermal headroom into transparent workload placement.
- **Simulation Scope**: This application is a research/educational prototype utilizing physics-grounded synthetic telemetry. Production deployment would interface directly with IPMI/Redfish BMC telemetry and liquid cooling manifolds.

---

## ⚠️ Known Prototype Limitations (For Defense Q&A)

1. **Synthetic Telemetry**: Uses a mathematical thermodynamic equilibrium model rather than full 3D Computational Fluid Dynamics (CFD) or physical sensor streams.
2. **Simplified Network Topology**: Focuses primarily on rack thermal and compute constraints without modeling inter-rack NVLink/InfiniBand communication latencies.
3. **Equilibrium Temperature**: Models steady-state equilibrium temperature rather than transient seconds-level thermal capacitance curves.
