# HeatWise: Intelligent Thermal-Aware Workload Placement in AI Data Centers

**HeatWise** is a simulation-based thermal intelligence and workload placement validated prototype designed for high-density AI data centers. Traditional workload schedulers primarily consider computational availability and are blind to localized thermal congestion. HeatWise uses machine learning to predict post-placement equilibrium temperatures, evaluate thermal headroom, and make transparent, explainable placement decisions.

---

## ⚡ Quick Demo (One Command)

HeatWise includes a unified launcher that starts both the presentation landing page and the Streamlit backend with a single command:

```bash
./run_heatwise.sh
```

Then open:
**http://localhost:8000**

To stop all services cleanly, press `Ctrl+C`.

### Architecture & Service Roles
- **Landing Page &rarr; `http://localhost:8000`**: The presentation layer featuring a 3D thermal data center visualization, architecture overview, and interactive entrance.
- **HeatWise Application &rarr; `http://localhost:8501`**: The actual working HeatWise prototype providing interactive thermal simulation, ML post-placement temperature inference, real-time rack evaluation, and scheduler benchmarks.

---

## 🚀 Manual Local Development

If you prefer running services individually:

### Prerequisites & Installation
```bash
pip install -r requirements.txt
```

### 1. Presentation Landing Page (Frontend)
```bash
python3 -m http.server 8000
```
Open: `http://localhost:8000`

### 2. HeatWise Application (Backend / Core)
In a separate terminal:
```bash
streamlit run app.py --server.port 8501
```
Open: `http://localhost:8501`

Clicking **Launch App &rarr;** or **Launch Interactive Demo &rarr;** on the landing page will open `http://localhost:8501`.

---

## 🌐 Hosted Demo & Application URL Configuration

When hosted online, the landing page CTAs can route to your hosted Streamlit instance instead of localhost. HeatWise supports a zero-build configuration hook via `window.HEATWISE_APP_URL`.

### How to Configure Hosted URL:
In `index.html`, set `window.HEATWISE_APP_URL` to your hosted application URL:
```html
<script>
    window.HEATWISE_APP_URL = "https://heatwise-app.yourdomain.com";
</script>
```

- **Local Fallback**: If `window.HEATWISE_APP_URL` is not set, all CTAs (`Launch App`, `Launch Interactive Demo`, `Launch the App`) automatically fall back to `http://localhost:8501`.
- **Architecture**: No build tools or bundlers are required. The implementation remains 100% vanilla HTML, CSS, and JavaScript.

---

## Architecture

The HeatWise system architecture follows a clear pipeline separating presentation, logic, prediction, and physical simulation:

```text
Streamlit UI
     ↓
HeatWise Decision Engine
     ↓
ML Temperature Predictor
     ↓
Thermal Simulation
```

- **Streamlit UI**: Provides the interactive frontend, visualization of the data center state, and demonstration workflows.
- **HeatWise Decision Engine**: Combines resource availability, ML predictions, and multiple heuristics to score and rank candidate racks.
- **ML Temperature Predictor**: A Random Forest model trained on synthetic telemetry to predict post-placement equilibrium temperatures based on rack power and cooling variables.
- **Thermal Simulation**: A reduced-order physics model that generates the thermal environment and ground truth for the 12 racks.

---

## Experimental methodology

The system evaluates thermal-aware scheduling using the following methodology:

- **12-rack simulation**: The data center is modeled as a 3x4 grid of high-density AI racks, incorporating localized thermal recirculation and cooling constraints.
- **Workload profiles**: Incoming jobs are categorized by intensity and resource demand (e.g., AI Training, LLM Inference) affecting both GPU usage and dynamic power consumption.
- **Scheduler baselines**: HeatWise is compared against traditional thermal-blind schedulers: **Round Robin** and **Least Loaded**.
- **Synthetic telemetry**: The simulation generates realistic telemetry features including inlet/outlet temperatures, power draw, and cooling efficiency based on physics-grounded mathematical approximations.
- **Train/test methodology**: The ML predictor is trained on thousands of samples generated from the thermal simulation with an 80/20 train/test split.
- **Thermal metrics**: Success is evaluated based on Peak Temperature, Thermal Violations (hotspots), Average Thermal Headroom, and Placement Failures.
- **Benchmark methodology**: A batch simulation stream of identical incoming workloads is processed sequentially by each scheduler to evaluate long-term thermal stability and failure rates.

---

## Limitations

This is a research prototype and college demonstration system. The following limitations apply:

1. **Reduced-order thermal model**: Heat is approximated using mathematical equilibrium formulas, not rigorous thermodynamics.
2. **Synthetic telemetry**: The sensor data is generated algorithmically and does not represent a physical data center.
3. **Aggregated neighbor features**: Spatial heat recirculation is simplified to aggregate neighboring temperatures.
4. **Tick-based workload duration**: Workloads do not have real-time durations, but rather operate on simulated steps or ticks.
5. **No CFD validation**: The thermal interactions have not been validated against Computational Fluid Dynamics (CFD) software.
6. **No real data-center telemetry**: The ML model was not trained on real-world BMC/IPMI or Redfish sensor streams.

*Note: HeatWise does not make real-world performance claims. It serves to visually demonstrate the conceptual benefit of thermal-aware scheduling.*
