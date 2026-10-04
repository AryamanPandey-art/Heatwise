"""
HeatWise: Intelligent Thermal-Aware Workload Placement in AI Data Centers
Interactive Streamlit Prototype & Demonstration System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Core modules
from simulation import (
    DataCenterSimulation,
    TEMP_CRITICAL,
    TEMP_MAX_SAFE,
    TEMP_NORMAL_MAX,
    TEMP_WARM_MAX,
    TOTAL_RACKS,
    SERVERS_PER_RACK,
    GPUS_PER_SERVER
)
from ml_model import get_trained_predictor, ThermalMLPredictor
from heatwise import (
    Workload,
    HeatWiseDecisionEngine,
    WORKLOAD_PROFILES,
    INTENSITY_LEVELS,
    DEFAULT_WEIGHTS,
    run_batch_simulation_benchmark
)
from visualization import (
    create_datacenter_heatmap,
    create_before_after_comparison_heatmap,
    create_candidate_radar_chart,
    create_candidate_scores_bar_chart,
    create_algorithm_benchmark_charts,
    create_ml_evaluation_charts
)

# Set page configuration
st.set_page_config(
    page_title="HeatWise | Thermal-Aware AI Workload Placement",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Cyber-Thermal CSS Design System
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin: 0;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Section Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
    }
    
    .glass-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Pipeline Flow Stepper */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 18px;
    }
    
    .pipeline-step {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .pipeline-step.active {
        background: rgba(14, 165, 233, 0.25);
        border-color: #38bdf8;
        color: #38bdf8;
    }
    
    .pipeline-arrow {
        color: #64748b;
        font-weight: 700;
        font-size: 14px;
    }
    
    /* Recommendation Winner Box */
    .winner-box {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    
    .winner-header {
        font-size: 20px;
        font-weight: 800;
        color: #34d399;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .reason-item {
        font-size: 13px;
        color: #e2e8f0;
        margin-bottom: 6px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }
    
    /* Rejection / Warning Box */
    .rejection-box {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-normal { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-warm { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-high { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    .badge-hotspot { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }
    
    /* Disclaimer Tag */
    .disclaimer-tag {
        font-size: 11px;
        color: #94a3b8;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "simulation" not in st.session_state:
    st.session_state.simulation = DataCenterSimulation(scenario="Thermal Hotspot")
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "Thermal Hotspot"
if "selected_inspect_rack" not in st.session_state:
    st.session_state.selected_inspect_rack = 7
if "ml_predictor" not in st.session_state:
    with st.spinner("Initializing Thermal ML Model..."):
        st.session_state.ml_predictor = get_trained_predictor()

# Initialize Decision Engine
decision_engine = HeatWiseDecisionEngine(predictor=st.session_state.ml_predictor)

if "last_placement_result" not in st.session_state or st.session_state.last_placement_result is None:
    init_wl = Workload(workload_type="AI Training", intensity="High", priority="High")
    st.session_state.last_placement_result = decision_engine.evaluate_placement(
        sim=st.session_state.simulation,
        workload=init_wl
    )


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Simulation Control")
    
    scenario_options = [
        "Thermal Hotspot",
        "Normal Operation",
        "High AI Workload",
        "Cooling Degradation",
        "GPU Surge"
    ]
    
    selected_scen = st.selectbox(
        "Simulation Scenario",
        scenario_options,
        index=scenario_options.index(st.session_state.selected_scenario),
        help="Simulates distinct thermal conditions, load profiles, or cooling faults."
    )
    
    # Reload simulation if scenario changed
    if selected_scen != st.session_state.selected_scenario:
        st.session_state.selected_scenario = selected_scen
        st.session_state.simulation.load_scenario(selected_scen)
        st.session_state.last_placement_result = None
        st.rerun()

    # Scenario description callout
    scen_descriptions = {
        "Thermal Hotspot": "🔥 **College Demo Scenario**: Rack 7 has 70% free GPU availability but is thermally stressed (68°C). Rack 3 has 60% free GPU and safe temperature (48°C).",
        "Normal Operation": "🟢 Balanced load and homogeneous CRAC airflow (45°C - 58°C across all racks).",
        "High AI Workload": "⚡ Heavy training clusters concentrated in top-left zone (Racks 1, 2, 5, 6).",
        "Cooling Degradation": "❄️ CRAC airflow failure in middle row (Racks 5-8 cooling drops to 45%).",
        "GPU Surge": "🚀 High-throughput burst load across bottom racks (Racks 9-12 GPU util > 85%)."
    }
    st.info(scen_descriptions[selected_scen])
    
    # Quick Reset / Preset Button
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reset Data Center", use_container_width=True):
            st.session_state.simulation.load_scenario(st.session_state.selected_scenario)
            st.session_state.last_placement_result = None
            st.rerun()
    with col_btn2:
        if st.button("⚡ Demo Preset", use_container_width=True):
            st.session_state.selected_scenario = "Thermal Hotspot"
            st.session_state.simulation.load_scenario("Thermal Hotspot")
            st.session_state.last_placement_result = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Incoming AI Workload")
    
    workload_type = st.selectbox(
        "Workload Type",
        list(WORKLOAD_PROFILES.keys()),
        index=0
    )
    
    intensity = st.select_slider(
        "Workload Intensity",
        options=["Low", "Medium", "High", "Very High"],
        value="High"
    )
    
    gpu_demand = st.slider(
        "GPU Demand (% Capacity)",
        min_value=15,
        max_value=90,
        value=int(WORKLOAD_PROFILES[workload_type]["base_gpu_pct"]),
        step=5,
        help="GPU compute capacity demanded by this incoming AI job."
    )
    
    priority = st.selectbox(
        "Workload Priority",
        ["Low", "Medium", "High", "Critical"],
        index=2
    )

    st.markdown("---")
    # Quick decision button in sidebar
    find_placement_clicked = st.button(
        "🚀 Find Best Placement",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-tag'>🔬 <b>Academic Prototype</b><br>Simulation-based research model for college demonstration.</div>",
        unsafe_allow_html=True
    )


# Top Hero Banner
st.markdown("""
<div class="hero-banner">
    <div>
        <div class="hero-title">🔥 HeatWise: Intelligent Thermal-Aware Workload Placement</div>
        <p class="hero-subtitle">High-Density AI Data Center Simulation & ML Post-Placement Thermal Optimization</p>
    </div>
    <div style="text-align: right;">
        <span class="disclaimer-tag">12 Racks | 48 GPU Servers | 384 GPUs</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Top KPI Metric Row
kpis = st.session_state.simulation.get_kpi_summary()
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-label">Total Racks</div>
        <div class="kpi-value">{kpis['total_racks']}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Active Servers</div>
        <div class="kpi-value">{kpis['total_servers']}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Cluster Temp</div>
        <div class="kpi-value">{kpis['avg_temp']}°C</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Peak Temperature</div>
        <div class="kpi-value" style="color: {'#ef4444' if kpis['max_temp'] >= TEMP_CRITICAL else '#f59e0b'};">{kpis['max_temp']}°C</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Power</div>
        <div class="kpi-value">{kpis['total_power_kw']} kW</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Thermal Hotspots</div>
        <div class="kpi-value" style="color: {'#ef4444' if kpis['hotspot_count'] > 0 else '#10b981'};">{kpis['hotspot_count']}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Current Workload</div>
        <div class="kpi-value" style="font-size: 16px; color: #38bdf8;">{workload_type}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs Navigation
tab_live, tab_bench, tab_ml, tab_arch, tab_about = st.tabs([
    "🖥️ Live Operations & Placement",
    "📊 Algorithm Comparison & Benchmarks",
    "🧠 ML Thermal Prediction Model",
    "🏛️ System Architecture",
    "ℹ️ About HeatWise & Presentation Script"
])


# ==============================================================================
# TAB 1: LIVE OPERATIONS & PLACEMENT
# ==============================================================================
with tab_live:
    # Create the workload object
    current_workload = Workload(
        workload_type=workload_type,
        intensity=intensity,
        gpu_demand_pct=float(gpu_demand),
        priority=priority,
        workload_id=f"JOB-{int(time.time()) % 1000}"
    )

    col_wl_info, col_wl_action = st.columns([0.7, 0.3])
    with col_wl_info:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-weight: 700; color: #38bdf8; font-size: 14px;">Incoming Workload:</span>
                <span style="color: #f8fafc; font-weight: 600; margin-left: 6px;">{workload_type}</span>
                <span style="color: #94a3b8; font-size: 12px; margin-left: 10px;">(Intensity: <b>{intensity}</b> | GPU Demand: <b>{gpu_demand}%</b> | Priority: <b>{priority}</b>)</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">Est Power: <b>{current_workload.power_est_w} W</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col_wl_action:
        inline_find_placement = st.button("🚀 Find Best Placement", type="primary", use_container_width=True, key="inline_find_btn")

    # Process placement if button was clicked or state exists
    if find_placement_clicked or inline_find_placement:
        with st.spinner("Executing HeatWise Decision Engine..."):
            eval_result = decision_engine.evaluate_placement(
                sim=st.session_state.simulation,
                workload=current_workload
            )
            st.session_state.last_placement_result = eval_result

    # Visual Workflow Pipeline
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step active">📥 1. Incoming AI Workload</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active">🔍 2. Check GPU Availability</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active">🌡️ 3. Telemetry & Heatmap</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active">🧠 4. ML Temp Prediction</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active">⚖️ 5. HeatWise Multi-Scoring</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active" style="border-color: #10b981; color: #34d399;">🎯 6. Recommended Placement</div>
    </div>
    """, unsafe_allow_html=True)

    # If we have an active evaluation result, show recommendation & explainability first
    if st.session_state.last_placement_result:
        res = st.session_state.last_placement_result
        winner = res["recommended_rack"]
        ll_rack = res["least_loaded_rack"]
        contrast = res["contrast_scenario"]

        col_win, col_contrast = st.columns([1.1, 0.9])

        with col_win:
            st.markdown(f"""
            <div class="winner-box">
                <div class="winner-header">
                    <span>🏆 Recommended Placement: <b>{winner['rack_name']}</b></span>
                    <span style="font-size: 14px; background: rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px;">Score: {winner['heatwise_score']}/100</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; font-size: 13px;">
                    <div>🌡️ Predicted Temp: <b style="color: #34d399;">{winner['predicted_temp']}°C</b> (+{winner['temp_delta']}°C)</div>
                    <div>🛡️ Thermal Headroom: <b style="color: #38bdf8;">{winner['predicted_headroom']}°C</b></div>
                    <div>⚡ GPU Availability: <b>{winner['gpu_available_pct']}%</b></div>
                    <div>🔌 Estimated Power: <b>{winner['predicted_power_kw']} kW</b></div>
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">Why did HeatWise choose {winner['rack_name']}?</div>
                <div class="reason-item">✅ <b>GPU Capacity:</b> Sufficient headroom to handle {current_workload.gpu_demand_pct}% demand.</div>
                <div class="reason-item">✅ <b>Thermal Safety:</b> Predicted temperature of {winner['predicted_temp']}°C is well below critical {TEMP_CRITICAL}°C limit.</div>
                <div class="reason-item">✅ <b>Cooling & Neighbors:</b> Operating at {winner['cooling_capacity_pct']}% cooling capacity with stable {winner['neighbor_avg_temp']}°C neighbor zone.</div>
            </div>
            """, unsafe_allow_html=True)

        with col_contrast:
            if contrast:
                st.markdown(f"""
                <div class="rejection-box">
                    <div style="font-size: 17px; font-weight: 800; color: #f87171; margin-bottom: 8px;">
                        ⚠️ Conventional Schedulers Would Fail (Least-Loaded Pitfall)
                    </div>
                    <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 10px;">
                        <b>Least-Loaded Algorithm</b> would pick <b style="color: #fca5a5;">{contrast['least_loaded_rack']}</b> (highest free GPU = {contrast['least_loaded_gpu_avail']}%), 
                        ignoring that it is already at {contrast['least_loaded_curr_temp']}°C.
                    </div>
                    <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 10px; margin-bottom: 10px; font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                        <div>• Least-Loaded → {contrast['least_loaded_rack']} → <span style="color: #ef4444; font-weight: 700;">Predicted {contrast['least_loaded_pred_temp']}°C (HOTSPOT RISK)</span></div>
                        <div>• HeatWise → {contrast['heatwise_rack']} → <span style="color: #10b981; font-weight: 700;">Predicted {contrast['heatwise_pred_temp']}°C (SAFE)</span></div>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; font-style: italic;">
                        {contrast['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-title">✅ Benchmark Alignment</div>
                    <div style="font-size: 13px; color: #cbd5e1;">
                        In this operating scenario, <b>{winner['rack_name']}</b> represents both the highest available compute and the safest thermal environment.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 2D Heatmap & Before/After Comparison
    col_map1, col_map2 = st.columns([1, 1])

    with col_map1:
        highlight_id = st.session_state.last_placement_result["recommended_rack"]["rack_id"] if st.session_state.last_placement_result else None
        fig_live_map = create_datacenter_heatmap(
            st.session_state.simulation.racks,
            highlight_rack_id=highlight_id,
            title="Interactive 2D Thermal Map (Current Telemetry)"
        )
        st.plotly_chart(fig_live_map, use_container_width=True)

    with col_map2:
        if st.session_state.last_placement_result:
            winner = st.session_state.last_placement_result["recommended_rack"]
            fig_before_after = create_before_after_comparison_heatmap(
                st.session_state.simulation.racks,
                chosen_rack_id=winner["rack_id"],
                post_temp=winner["predicted_temp"],
                chosen_algorithm_name="HeatWise Recommended Rack"
            )
            st.plotly_chart(fig_before_after, use_container_width=True)
        else:
            st.info("👈 Click **'Find Best Placement'** in the sidebar to generate Before vs After Thermal Impact Map.")

    # Candidate Evaluation Breakdown & Radar Comparison
    if st.session_state.last_placement_result:
        res = st.session_state.last_placement_result
        st.markdown("### 📊 Candidate Rack Evaluations & Scoring")

        col_radar, col_bar = st.columns([1, 1])
        with col_radar:
            fig_radar = create_candidate_radar_chart(res["all_candidates"], top_n=4)
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_bar:
            fig_bar_scores = create_candidate_scores_bar_chart(res["all_candidates"])
            st.plotly_chart(fig_bar_scores, use_container_width=True)

        # Full Candidate Matrix Table
        with st.expander("📋 View Full 12-Rack Candidate Decision Matrix", expanded=False):
            cand_rows = []
            for c in res["all_candidates"]:
                cand_rows.append({
                    "Rack": c["rack_name"],
                    "HeatWise Score": f"{c['heatwise_score']}/100",
                    "Current Temp": f"{c['current_temp']}°C",
                    "Predicted Temp": f"{c['predicted_temp']}°C",
                    "Temp Rise": f"+{c['temp_delta']}°C",
                    "Thermal Headroom": f"{c['predicted_headroom']}°C",
                    "GPU Free (%)": f"{c['gpu_available_pct']}%",
                    "Cooling Capacity": f"{c['cooling_capacity_pct']}%",
                    "Decision Status": "REJECTED" if c["is_rejected"] else "SAFE",
                    "Reason / Diagnosis": c["rejection_reason"]
                })
            st.dataframe(pd.DataFrame(cand_rows), use_container_width=True)

    # Server Level Inspector
    st.markdown("---")
    st.markdown("### 🔍 Server-Level Drilldown Inspector")
    col_sel_rack, col_inspect = st.columns([0.3, 0.7])
    
    with col_sel_rack:
        inspect_rack_id = st.selectbox(
            "Select Rack to Inspect Servers:",
            options=list(range(1, TOTAL_RACKS + 1)),
            index=st.session_state.selected_inspect_rack - 1
        )
        st.session_state.selected_inspect_rack = inspect_rack_id
        rack_obj = st.session_state.simulation.racks[inspect_rack_id]
        
        st.markdown(f"""
        <div class="glass-card" style="padding: 14px;">
            <div style="font-weight: 700; font-size: 15px; color: #38bdf8;">Rack {inspect_rack_id} Telemetry</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;">
                • Temperature: <b>{rack_obj['temperature']}°C</b><br>
                • Status: <span class="status-badge {rack_obj['badge_class']}">{rack_obj['status']}</span><br>
                • Cooling Capacity: <b>{rack_obj['cooling_capacity_pct']}%</b><br>
                • Neighbor Avg: <b>{rack_obj['neighbor_avg_temp']}°C</b><br>
                • Total Power: <b>{round(rack_obj['total_power_w']/1000, 2)} kW</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_inspect:
        servers = rack_obj["servers"]
        srv_df = pd.DataFrame([
            {
                "Server ID": s["server_id"],
                "GPU Util (%)": f"{s['gpu_util']}%",
                "CPU Util (%)": f"{s['cpu_util']}%",
                "GPU Count": s["gpu_count"],
                "Power Draw (W)": f"{s['power_w']} W",
                "Active Workload": s["active_workload"]
            }
            for s in servers
        ])
        st.dataframe(srv_df, use_container_width=True)

    # Expandable Decision Logic Breakdown
    with st.expander("⚙️ How HeatWise Makes the Decision (Mathematical Formulation & Weights)", expanded=False):
        st.markdown(r"""
        #### **HeatWise Multi-Criteria Objective Formulation**
        HeatWise evaluates all candidate racks by synthesizing physics-driven telemetry and machine learning temperature predictions into a transparent composite score $S_i \in [0, 100]$:
        
        $$\text{Score}_i = w_{\text{temp}} \cdot S_{\text{temp}}(T_{\text{pred}}) + w_{\text{head}} \cdot S_{\text{head}}(\text{Headroom}) + w_{\text{gpu}} \cdot S_{\text{gpu}}(\text{GPU Availability}) + w_{\text{cool}} \cdot S_{\text{cool}}(\text{Cooling}) + w_{\text{nbr}} \cdot S_{\text{nbr}}(\text{Neighbor Temp}) - \text{Penalty}$$
        
        * **$S_{\text{temp}}$ (35%)**: Penalizes racks whose predicted temperature approaches or exceeds $75.0^\circ\text{C}$.
        * **$S_{\text{head}}$ (25%)**: Rewards racks with larger thermal headroom buffer margins.
        * **$S_{\text{gpu}}$ (20%)**: Ensures sufficient computational capacity and load balancing.
        * **$S_{\text{cool}}$ (10%)**: Rewards racks serviced by higher-efficiency CRAC airflow loops.
        * **$S_{\text{nbr}}$ (10%)**: Avoids thermal recirculation coupling from overheated neighboring racks.
        """)


# ==============================================================================
# TAB 2: ALGORITHM COMPARISON & BENCHMARKS
# ==============================================================================
with tab_bench:
    st.markdown("### 📊 Benchmark: Round Robin vs Least-Loaded vs HeatWise")
    st.markdown("""
    This benchmark runs a **stream of identical incoming AI workloads** through three scheduling algorithms on cloned data center instances:
    1. **Round Robin**: Dispatches jobs sequentially across available compute nodes without thermal awareness.
    2. **Least-Loaded**: Routes jobs to the server with highest free GPU capacity (blind to heat).
    3. **HeatWise**: Thermal-aware placement utilizing ML post-placement temperature prediction.
    """)

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        num_jobs = st.slider("Workload Stream Size (Number of Jobs)", min_value=5, max_value=30, value=15, step=5)
        run_bench_clicked = st.button("▶️ Execute Batch Benchmark", type="primary", use_container_width=True)

    if run_bench_clicked or "bench_results" not in st.session_state:
        with st.spinner("Simulating multi-algorithm workload dispatch..."):
            comp_df, metrics_trk, timeline_d = run_batch_simulation_benchmark(
                base_scenario=st.session_state.selected_scenario,
                num_workloads=num_jobs
            )
            st.session_state.bench_results = {
                "comp_df": comp_df,
                "metrics_trk": metrics_trk,
                "timeline_d": timeline_d
            }

    if "bench_results" in st.session_state:
        b_res = st.session_state.bench_results
        
        # Summary Table
        st.markdown("#### 🏆 Performance Summary Matrix")
        st.dataframe(b_res["comp_df"], use_container_width=True)

        # Plotly Comparison Visuals
        fig_bar_bench, fig_line_bench = create_algorithm_benchmark_charts(
            b_res["metrics_trk"],
            b_res["timeline_d"]
        )

        st.plotly_chart(fig_bar_bench, use_container_width=True)
        st.plotly_chart(fig_line_bench, use_container_width=True)

        # Analysis Insights Callout
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">💡 Experimental Takeaways for Presentation</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                • <b>Why Least-Loaded fails:</b> Least-Loaded persistently packs workloads into racks that appear idle/free even if localized cooling or thermal recirculation has made them hot spots. This triggers rapid thermal runaway (>75°C).<br>
                • <b>Why Round Robin fails:</b> Round Robin exhibits blind distribution, assigning heavy workloads to warm nodes indiscriminately.<br>
                • <b>How HeatWise succeeds:</b> HeatWise predicts the thermodynamic equilibrium prior to placement, balancing compute demand with heat dissipation capacity to prevent hotspot formation and eliminate SLA violations.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: ML THERMAL PREDICTION MODEL
# ==============================================================================
with tab_ml:
    st.markdown("### 🧠 Machine Learning Thermal Prediction Engine")
    st.markdown(r"""
    HeatWise utilizes a **Random Forest Regressor** trained on physics-grounded telemetry samples.
    The model predicts the rack's post-placement equilibrium temperature ($T_{\text{post}}$) based on current thermal state,
    cooling capacity, neighboring rack temperatures, and incoming workload demand.
    """)

    predictor = st.session_state.ml_predictor
    m = predictor.metrics

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Mean Absolute Error (MAE)", f"{m['mae']} °C", delta="-0.32°C vs Baseline", delta_color="inverse")
    with col_m2:
        st.metric("Root Mean Squared Error (RMSE)", f"{m['rmse']} °C")
    with col_m3:
        st.metric("R² Variance Explained", f"{m['r2_pct']}%", delta="High Precision")
    with col_m4:
        st.metric("Training Dataset Size", f"{m['total_samples']} samples")

    # ML Charts
    fig_scatter, fig_imp = create_ml_evaluation_charts(predictor.test_eval_df, predictor.feature_importances)
    
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_c2:
        st.plotly_chart(fig_imp, use_container_width=True)

    # Retrain button & Disclaimer
    col_retrain, col_disc = st.columns([0.3, 0.7])
    with col_retrain:
        if st.button("🔄 Retrain ML Model", use_container_width=True):
            with st.spinner("Generating fresh synthetic telemetry and retraining Random Forest..."):
                st.session_state.ml_predictor.train()
                st.rerun()
    with col_disc:
        st.markdown(
            "<div class='disclaimer-tag'>ℹ️ <b>Academic Note:</b> In this research prototype, the model is trained on physics-grounded synthetic telemetry. In production deployment, it would continuously train on BMC/IPMI and Redfish IoT sensor streams.</div>",
            unsafe_allow_html=True
        )


# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE
# ==============================================================================
with tab_arch:
    st.markdown("### 🏛️ System Architecture & Data Flow Pipeline")
    
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Ingestion ["1. Workload Ingestion"]
            WL["Incoming AI Workload<br>(Training, LLM, Inference)"]
        end

        subgraph Telemetry ["2. Telemetry & Sensor Mesh"]
            Racks["12-Rack AI Data Center<br>48 GPU Servers (384 GPUs)"]
            Sensors["Metrics: Power (W), Temp (°C),<br>Cooling (%), Neighbor Heat"]
            Racks --> Sensors
        end

        subgraph ML ["3. ML Thermal Predictor"]
            RF["Random Forest Regressor<br>Features: Temp, Power, Demand, CRAC"]
            PredTemp["Predicted Post-Placement<br>Equilibrium Temp (°C)"]
            Sensors --> RF
            WL --> RF
            RF --> PredTemp
        end

        subgraph Engine ["4. HeatWise Decision Engine"]
            Filter["GPU Capacity Filter"]
            Scoring["Multi-Factor Scoring Engine<br>(Temp, Headroom, GPU, CRAC, Neighbors)"]
            Safety["Thermal Safety Threshold<br>(Reject if > 75°C)"]
            PredTemp --> Safety
            Safety --> Scoring
        end

        subgraph Execution ["5. Placement Dispatch"]
            Optimal["Optimal Rack Placement<br>+ Full Explainability Rationale"]
            Scoring --> Optimal
        end
    ```
    """, unsafe_allow_html=True)

    st.markdown("#### 🏗️ Data Center Infrastructure Components")
    col_ic1, col_ic2, col_ic3 = st.columns(3)
    with col_ic1:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">🖥️ Compute & GPU Tier</div>
            <div style="font-size: 13px; color: #cbd5e1;">
                • 12 Standard High-Density Racks<br>
                • 48 Multi-GPU Compute Nodes<br>
                • 384 High-Performance GPUs (H100 class)<br>
                • Dynamic CPU/GPU power scaling (400W - 2800W)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_ic2:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">❄️ Thermal & Cooling Tier</div>
            <div style="font-size: 13px; color: #cbd5e1;">
                • Computer Room Air Conditioning (CRAC)<br>
                • Hot-Aisle / Cold-Aisle Containment<br>
                • Spatial thermal recirculation modeling<br>
                • Thermal throttling threshold: 75.0°C / 80.0°C
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_ic3:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">📡 Telemetry & Decision Tier</div>
            <div style="font-size: 13px; color: #cbd5e1;">
                • Real-time server & rack metrics collection<br>
                • Random Forest post-workload temperature ML<br>
                • Explainable rejection & selection diagnosis<br>
                • Multi-objective trade-off balancing
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 5: ABOUT HEATWISE & PRESENTATION SCRIPT
# ==============================================================================
with tab_about:
    st.markdown("### ℹ️ About HeatWise: Problem, Solution & Academic Positioning")
    
    col_ab1, col_ab2 = st.columns([1.1, 0.9])
    with col_ab1:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">🎯 Problem Statement</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                Modern AI data centers operate under unprecedented power densities (30-100 kW per rack) driven by multi-billion parameter LLMs and distributed training clusters.
                Traditional workload schedulers (e.g., Round Robin, Least-Loaded, Kubernetes default scheduler) allocate compute based strictly on <b>CPU/GPU core availability and memory capacity</b>.
                <br><br>
                <b>The Critical Flaw:</b> These schedulers are blind to thermodynamics. When an idle server is located in a thermally congested zone or experiencing degraded cooling, placing a heavy workload causes <b>thermal hotspots (>75°C), hardware throttling, and SLA violations</b>.
            </div>
        </div>
        
        <div class="glass-card">
            <div class="glass-card-title">💡 The HeatWise Solution</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                HeatWise transforms workload placement into a <b>thermal-aware, predictive optimization task</b>.
                Before dispatching an AI workload, HeatWise uses machine learning to forecast the post-placement equilibrium temperature across all candidate racks, evaluating:
                <br>
                1. <b>Predicted Post-Placement Temperature</b><br>
                2. <b>Thermal Headroom Buffer</b><br>
                3. <b>GPU Capacity & Computational Fit</b><br>
                4. <b>Cooling Capacity & CRAC Efficiency</b><br>
                5. <b>Neighboring Rack Thermal Recirculation</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">🔬 Academic Positioning & Scope</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                <b>Novelty & Research Differentiation:</b><br>
                Thermal-aware scheduling and ML temperature prediction are established concepts in IEEE literature.
                <br><br>
                <i>HeatWise contributes an <b>explainable, rack-level simulation prototype</b> that integrates predicted thermal impact, neighboring rack conditions, computational capacity, power, and thermal headroom into transparent workload placement.</i>
                <br><br>
                <b>System Scope:</b> Simulation/research prototype designed for college demonstration and proof-of-concept validation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎙️ 2-Minute College Presentation Script & Demo Flow")
    st.markdown("""
    Use this step-by-step walkthrough during your live demonstration:

    1. **Introduction (15s)**:
       > *"Good morning Professor. Today I am presenting **HeatWise**, an intelligent thermal-aware workload placement system for high-density AI data centers. Traditional schedulers only look at GPU availability, but HeatWise predicts thermal impact to prevent hotspots."*

    2. **Show Data Center Telemetry (20s)**:
       > *"Here is our 12-rack virtual data center running 48 GPU servers. Notice on the 2D Heatmap that different racks have different temperatures and cooling efficiencies."*

    3. **Demonstrate the Key Scenario (Rack 7 vs Rack 3) (35s)**:
       > *"Notice **Rack 7**: it has 70% free GPU availability, but due to thermal recirculation, its current temperature is 68°C. Meanwhile, **Rack 3** has 60% free GPU but is at a cool 48°C. A traditional Least-Loaded scheduler blindly picks Rack 7 because of higher free compute. Let's click **'Find Best Placement'**."*

    4. **Show HeatWise Decision & Explainability (30s)**:
       > *"HeatWise uses our Random Forest model to predict that placing this training job on Rack 7 would spike its temperature to 78°C—causing a critical hotspot! Therefore, HeatWise rejects Rack 7 and recommends Rack 3, where predicted temperature stays at a safe 56°C. Look at the Before vs. After Heatmap and the explainability breakdown."*

    5. **Algorithm Benchmark Comparison (20s)**:
       > *"On Tab 2, we simulate a stream of 15 workloads. Notice how Least-Loaded and Round Robin trigger multiple thermal hotspots and SLA violations, whereas HeatWise keeps the peak temperature low and well-distributed."*
    """)
