"""
HeatWise: Visualization Module
Provides interactive Plotly charts, 2D thermal heatmaps, before/after thermal comparisons,
radar evaluation charts, and algorithm benchmark visualizers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from simulation import GRID_ROWS, GRID_COLS, RACK_COORDS, TEMP_CRITICAL, TEMP_NORMAL_MAX, TEMP_WARM_MAX


# Dark Theme Colors for Data Center Aesthetics
DARK_THEME_TEMPLATE = "plotly_dark"
COLOR_BG = "rgba(15, 23, 42, 0.95)"
COLOR_PAPER = "rgba(15, 23, 42, 0.0)"
COLOR_TEXT = "#f8fafc"
COLOR_GRID = "rgba(51, 65, 85, 0.4)"

# Custom Thermal Colorscale
THERMAL_COLORSCALE = [
    [0.00, "#059669"],  # 35°C Safe Green
    [0.35, "#10b981"],  # 50°C Emerald
    [0.55, "#eab308"],  # 62°C Warning Yellow
    [0.72, "#f97316"],  # 68°C Orange
    [0.85, "#ef4444"],  # 75°C Hotspot Crimson
    [1.00, "#991b1b"]   # 85°C Critical Dark Red
]


def create_datacenter_heatmap(
    sim_racks: Dict[int, Dict[str, Any]],
    highlight_rack_id: Optional[int] = None,
    title: str = "Live Data Center Thermal Map (12 Racks)"
) -> go.Figure:
    """
    Creates a 2D interactive 3x4 heatmap of the data center temperature distribution.
    """
    z_matrix = np.zeros((GRID_ROWS, GRID_COLS))
    text_matrix = [["" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    hover_matrix = [["" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for r_id, r_data in sim_racks.items():
        row = r_data["row"]
        col = r_data["col"]
        temp = r_data["temperature"]
        z_matrix[row][col] = temp

        status_icon = "🟢" if temp < TEMP_NORMAL_MAX else ("🟡" if temp < TEMP_WARM_MAX else ("🟠" if temp < TEMP_CRITICAL else "🔴"))
        highlight_star = " ⭐ [RECOMMENDED]" if highlight_rack_id == r_id else ""

        text_matrix[row][col] = (
            f"<b>Rack {r_id}</b> {status_icon}{highlight_star}<br>"
            f"<b>{temp:.1f}°C</b><br>"
            f"<span style='font-size:11px;'>GPU Free: {r_data['gpu_available_pct']}% | {round(r_data['total_power_w']/1000, 1)}kW</span>"
        )

        hover_matrix[row][col] = (
            f"<b>Rack {r_id} ({r_data['status']})</b><br>"
            f"Temperature: {temp}°C<br>"
            f"GPU Availability: {r_data['gpu_available_pct']}%<br>"
            f"Thermal Headroom: {r_data['thermal_headroom']}°C<br>"
            f"Total Power: {round(r_data['total_power_w']/1000, 2)} kW<br>"
            f"Cooling Capacity: {r_data['cooling_capacity_pct']}%<br>"
            f"Neighbor Avg Temp: {r_data['neighbor_avg_temp']}°C<br>"
            f"Primary Workload: {r_data['primary_workload']}"
        )

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=["Col 1", "Col 2", "Col 3", "Col 4"],
        y=["Row 1", "Row 2", "Row 3"],
        text=text_matrix,
        hovertext=hover_matrix,
        hoverinfo="text",
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#ffffff", "family": "Inter, sans-serif"},
        colorscale=THERMAL_COLORSCALE,
        zmin=38.0,
        zmax=82.0,
        colorbar=dict(
            title=dict(text="Temp (°C)", font=dict(color="#f8fafc", size=13)),
            tickfont=dict(color="#f8fafc"),
            tickvals=[40, 50, 60, 70, 75, 80],
            ticktext=["40°C", "50°C", "60°C (Warm)", "70°C", "75°C (Hotspot)", "80°C (Limit)"],
            len=0.9,
            thickness=16
        )
    ))

    # Add highlight border if specified
    shapes = []
    if highlight_rack_id in sim_racks:
        h_row = sim_racks[highlight_rack_id]["row"]
        h_col = sim_racks[highlight_rack_id]["col"]
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=h_col - 0.48, y0=h_row - 0.48,
            x1=h_col + 0.48, y1=h_row + 0.48,
            line=dict(color="#38bdf8", width=4.5),
            fillcolor="rgba(56, 189, 248, 0.15)"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f8fafc", size=17, family="Inter, sans-serif")),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=40, r=40, t=55, b=30),
        height=380,
        shapes=shapes,
        yaxis=dict(autorange="reversed", tickfont=dict(color="#cbd5e1", size=12)),
        xaxis=dict(tickfont=dict(color="#cbd5e1", size=12))
    )

    return fig


def create_before_after_comparison_heatmap(
    sim_racks: Dict[int, Dict[str, Any]],
    chosen_rack_id: int,
    post_temp: float,
    chosen_algorithm_name: str = "HeatWise Recommendation"
) -> go.Figure:
    """
    Creates a side-by-side Before Placement vs After Placement thermal heatmap.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>BEFORE Placement</b> (Current State)", f"<b>AFTER Placement</b> ({chosen_algorithm_name})"),
        horizontal_spacing=0.10
    )

    # Matrix for Before
    z_before = np.zeros((GRID_ROWS, GRID_COLS))
    text_before = [["" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    # Matrix for After
    z_after = np.zeros((GRID_ROWS, GRID_COLS))
    text_after = [["" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for r_id, r_data in sim_racks.items():
        row, col = r_data["row"], r_data["col"]
        t_curr = r_data["temperature"]
        z_before[row][col] = t_curr

        s_icon_b = "🟢" if t_curr < TEMP_NORMAL_MAX else ("🟡" if t_curr < TEMP_WARM_MAX else ("🟠" if t_curr < TEMP_CRITICAL else "🔴"))
        text_before[row][col] = f"Rack {r_id} {s_icon_b}<br><b>{t_curr:.1f}°C</b>"

        # After state: target rack updated with post_temp
        t_after = post_temp if r_id == chosen_rack_id else t_curr
        z_after[row][col] = t_after
        s_icon_a = "🟢" if t_after < TEMP_NORMAL_MAX else ("🟡" if t_after < TEMP_WARM_MAX else ("🟠" if t_after < TEMP_CRITICAL else "🔴"))
        target_tag = " 🎯 [PLACED]" if r_id == chosen_rack_id else ""
        text_after[row][col] = f"Rack {r_id} {s_icon_a}{target_tag}<br><b>{t_after:.1f}°C</b>"

    fig.add_trace(
        go.Heatmap(
            z=z_before,
            x=["Col 1", "Col 2", "Col 3", "Col 4"],
            y=["Row 1", "Row 2", "Row 3"],
            text=text_before,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#ffffff"),
            colorscale=THERMAL_COLORSCALE,
            zmin=38.0, zmax=82.0,
            showscale=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=z_after,
            x=["Col 1", "Col 2", "Col 3", "Col 4"],
            y=["Row 1", "Row 2", "Row 3"],
            text=text_after,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#ffffff"),
            colorscale=THERMAL_COLORSCALE,
            zmin=38.0, zmax=82.0,
            colorbar=dict(
                title="Temp (°C)",
                tickfont=dict(color="#f8fafc"),
                len=0.9,
                thickness=14
            )
        ),
        row=1, col=2
    )

    # Highlight placed rack on 'After' subplot
    t_row, t_col = sim_racks[chosen_rack_id]["row"], sim_racks[chosen_rack_id]["col"]
    fig.add_shape(
        type="rect",
        xref="x2", yref="y2",
        x0=t_col - 0.48, y0=t_row - 0.48,
        x1=t_col + 0.48, y1=t_row + 0.48,
        line=dict(color="#38bdf8", width=4),
        fillcolor="rgba(56, 189, 248, 0.15)"
    )

    fig.update_layout(
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=30, r=30, t=50, b=30),
        height=340
    )
    fig.update_yaxes(autorange="reversed")

    return fig


def create_candidate_radar_chart(candidates: List[Dict[str, Any]], top_n: int = 4) -> go.Figure:
    """
    Creates a multi-axis Radar Chart comparing top candidate racks across all evaluation metrics.
    """
    categories = [
        "Temp Safety",
        "Thermal Headroom",
        "GPU Availability",
        "Cooling Capacity",
        "Neighbor Stability"
    ]

    colors = ["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

    fig = go.Figure()

    for idx, c in enumerate(candidates[:top_n]):
        subs = c["sub_scores"]
        values = [
            max(0, subs["temp_safety"]),
            subs["thermal_headroom"],
            subs["gpu_fit"],
            subs["cooling_eff"],
            subs["neighbor_stability"]
        ]
        # Close radar polygon
        values.append(values[0])
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            name=f"{c['rack_name']} (Score: {c['heatwise_score']})",
            line=dict(color=colors[idx % len(colors)], width=2.5),
            fill='toself',
            fillcolor=f"rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.12)"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#94a3b8", size=10),
                gridcolor="rgba(148, 163, 184, 0.2)"
            ),
            angularaxis=dict(
                tickfont=dict(color="#f8fafc", size=11, family="Inter, sans-serif"),
                gridcolor="rgba(148, 163, 184, 0.2)"
            ),
            bgcolor="rgba(15, 23, 42, 0.4)"
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#f8fafc", size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5
        ),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        margin=dict(l=30, r=30, t=30, b=50),
        height=340
    )

    return fig


def create_candidate_scores_bar_chart(candidates: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a sorted bar chart of HeatWise composite scores across all 12 racks.
    """
    racks = [c["rack_name"] for c in candidates]
    scores = [c["heatwise_score"] for c in candidates]
    pred_temps = [c["predicted_temp"] for c in candidates]

    bar_colors = []
    for c in candidates:
        if c["is_rejected"]:
            bar_colors.append("#ef4444")
        elif c["heatwise_score"] >= 75.0:
            bar_colors.append("#10b981")
        elif c["heatwise_score"] >= 50.0:
            bar_colors.append("#38bdf8")
        else:
            bar_colors.append("#f59e0b")

    hover_texts = [
        f"<b>{c['rack_name']}</b><br>"
        f"HeatWise Score: {c['heatwise_score']}/100<br>"
        f"Predicted Temp: {c['predicted_temp']}°C<br>"
        f"GPU Available: {c['gpu_available_pct']}%<br>"
        f"Status: {'REJECTED (' + c['rejection_reason'] + ')' if c['is_rejected'] else 'Valid Candidate'}"
        for c in candidates
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=racks,
            y=scores,
            text=[f"{s:.1f}" for s in scores],
            textposition='auto',
            textfont=dict(color="#ffffff", size=11),
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            hovertext=hover_texts,
            hoverinfo="text"
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Candidate HeatWise Scores (Ranked)</b>", font=dict(color="#f8fafc", size=14)),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=30, r=20, t=40, b=30),
        height=280,
        yaxis=dict(
            title="Composite Score (0-100)",
            range=[0, 105],
            gridcolor=COLOR_GRID,
            tickfont=dict(color="#cbd5e1")
        ),
        xaxis=dict(tickfont=dict(color="#cbd5e1", size=11))
    )

    return fig


def create_algorithm_benchmark_charts(metrics_tracker: Dict[str, Any], timeline_data: List[Dict[str, Any]]) -> Tuple[go.Figure, go.Figure]:
    """
    Creates:
    1. Multi-metric comparison bar chart (Peak Temp, Hotspots, SLA Violations).
    2. Dynamic timeline trace showing peak data center temperature under each scheduler.
    """
    algos = ["Round Robin", "Least Loaded", "HeatWise"]
    colors = ["#f59e0b", "#ef4444", "#10b981"]

    # 1. Bar Chart Comparison
    fig_bar = make_subplots(
        rows=1, cols=3,
        subplot_titles=("<b>Peak Temperature (°C)</b>", "<b>Thermal Hotspots (≥75°C)</b>", "<b>SLA Violations</b>"),
        horizontal_spacing=0.08
    )

    # Peak Temp Subplot
    peak_temps = [metrics_tracker[a]["peak_temp"] for a in algos]
    fig_bar.add_trace(
        go.Bar(
            x=algos, y=peak_temps,
            marker_color=colors,
            text=[f"{v}°C" for v in peak_temps],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=1
    )

    # Hotspots Subplot
    hotspots = [metrics_tracker[a]["hotspots"] for a in algos]
    fig_bar.add_trace(
        go.Bar(
            x=algos, y=hotspots,
            marker_color=colors,
            text=[str(v) for v in hotspots],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=2
    )

    # SLA Violations Subplot
    sla_viols = [metrics_tracker[a]["sla_violations"] for a in algos]
    fig_bar.add_trace(
        go.Bar(
            x=algos, y=sla_viols,
            marker_color=colors,
            text=[str(v) for v in sla_viols],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=3
    )

    fig_bar.update_layout(
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=30, r=30, t=50, b=30),
        height=280
    )
    fig_bar.update_yaxes(gridcolor=COLOR_GRID)

    # 2. Timeline Progression Line Chart
    df_time = pd.DataFrame(timeline_data)
    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(
        x=df_time["Step"],
        y=df_time["Round Robin Max Temp"],
        mode="lines+markers",
        name="Round Robin",
        line=dict(color="#f59e0b", width=2.5, dash="dot"),
        marker=dict(size=6)
    ))

    fig_line.add_trace(go.Scatter(
        x=df_time["Step"],
        y=df_time["Least Loaded Max Temp"],
        mode="lines+markers",
        name="Least Loaded",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=7)
    ))

    fig_line.add_trace(go.Scatter(
        x=df_time["Step"],
        y=df_time["HeatWise Max Temp"],
        mode="lines+markers",
        name="HeatWise (Thermal-Aware)",
        line=dict(color="#10b981", width=3.5),
        marker=dict(size=8, symbol="diamond")
    ))

    # Add threshold line for Critical Hotspot (75°C)
    fig_line.add_hline(
        y=TEMP_CRITICAL,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Critical Hotspot Threshold (75°C)",
        annotation_position="bottom right",
        annotation_font=dict(color="#f87171", size=11)
    )

    fig_line.update_layout(
        title=dict(text="<b>Cluster Peak Temperature Trajectory Across Workload Stream</b>", font=dict(color="#f8fafc", size=15)),
        xaxis=dict(title="Workload Placement Step #", gridcolor=COLOR_GRID, tickfont=dict(color="#cbd5e1")),
        yaxis=dict(title="Max Cluster Temperature (°C)", gridcolor=COLOR_GRID, tickfont=dict(color="#cbd5e1")),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        legend=dict(font=dict(color="#f8fafc"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320
    )

    return fig_bar, fig_line


def create_ml_evaluation_charts(eval_sample_df: pd.DataFrame, feature_importances: Dict[str, float]) -> Tuple[go.Figure, go.Figure]:
    """
    Creates:
    1. Actual vs Predicted Scatter Plot with ideal fit line.
    2. Feature Importance Horizontal Bar Chart.
    """
    # 1. Scatter Plot
    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=eval_sample_df["Actual (°C)"],
        y=eval_sample_df["Predicted (°C)"],
        mode="markers",
        name="Predictions",
        marker=dict(color="#38bdf8", size=7, opacity=0.8, line=dict(color="#ffffff", width=0.5)),
        text=[f"Actual: {a}°C<br>Pred: {p}°C<br>Res: {r:+.2f}°C" for a, p, r in zip(eval_sample_df["Actual (°C)"], eval_sample_df["Predicted (°C)"], eval_sample_df["Residual (°C)"])],
        hoverinfo="text"
    ))

    min_val = min(eval_sample_df["Actual (°C)"].min(), eval_sample_df["Predicted (°C)"].min()) - 2
    max_val = max(eval_sample_df["Actual (°C)"].max(), eval_sample_df["Predicted (°C)"].max()) + 2

    # Ideal Reference Line y = x
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Fit (y=x)",
        line=dict(color="#10b981", width=2, dash="dash")
    ))

    fig_scatter.update_layout(
        title=dict(text="<b>Model Accuracy: Actual vs Predicted Post-Placement Temperature</b>", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(title="Actual Equilibrium Temperature (°C)", gridcolor=COLOR_GRID, range=[min_val, max_val]),
        yaxis=dict(title="ML Predicted Temperature (°C)", gridcolor=COLOR_GRID, range=[min_val, max_val]),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=40, r=30, t=45, b=40),
        height=320,
        legend=dict(font=dict(color="#f8fafc"))
    )

    # 2. Feature Importances
    feats = list(feature_importances.keys())[::-1]
    imps = list(feature_importances.values())[::-1]

    fig_imp = go.Figure(data=[
        go.Bar(
            x=imps,
            y=feats,
            orientation='h',
            marker=dict(
                color=imps,
                colorscale="Viridis",
                line=dict(color="rgba(255,255,255,0.15)", width=1)
            ),
            text=[f"{v:.1f}%" for v in imps],
            textposition="outside",
            textfont=dict(color="#f8fafc", size=11)
        )
    ])

    fig_imp.update_layout(
        title=dict(text="<b>Random Forest Feature Importance (%)</b>", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(title="Relative Importance (%)", gridcolor=COLOR_GRID, range=[0, max(imps) * 1.25]),
        yaxis=dict(tickfont=dict(color="#f8fafc", size=11)),
        template=DARK_THEME_TEMPLATE,
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        margin=dict(l=150, r=40, t=45, b=40),
        height=320
    )

    return fig_scatter, fig_imp
