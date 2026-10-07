"""
HeatWise UI Theme & Styling Module
Provides modern light & dark technical design systems, CSS injection, responsive card builders,
control-plane rack topology grids, and Plotly chart theming for localhost:8501.
"""

from typing import Dict, Any, List, Optional, Tuple
import textwrap
import plotly.graph_objects as go
from config import TEMP_CRITICAL, TEMP_MAX_SAFE, TEMP_NORMAL_MAX, TEMP_WARM_MAX

# ==============================================================================
# COLOR PALETTES & DESIGN TOKENS
# ==============================================================================

THEME_CONFIG = {
    "dark": {
        "bg_primary": "#080b11",
        "bg_secondary": "#0d131f",
        "card_bg": "#101726",
        "card_bg_elevated": "#151e32",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "card_border_highlight": "rgba(56, 189, 248, 0.4)",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#64748b",
        "accent_blue": "#38bdf8",
        "accent_indigo": "#818cf8",
        "accent_purple": "#c084fc",
        "accent_cyan": "#06b6d4",
        "accent_emerald": "#10b981",
        "accent_amber": "#f59e0b",
        "accent_rose": "#ef4444",
        "rack_frame": "#131b2c",
        "blade_bg": "#0c1322",
        "plotly_template": "plotly_dark",
        "plotly_paper": "rgba(0, 0, 0, 0)",
        "plotly_plot": "rgba(13, 19, 31, 0.6)",
        "plotly_grid": "rgba(255, 255, 255, 0.06)",
        "plotly_text": "#cbd5e1",
        "shadow": "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
        "badge_border": "rgba(255, 255, 255, 0.12)"
    },
    "light": {
        "bg_primary": "#f8fafc",
        "bg_secondary": "#ffffff",
        "card_bg": "#ffffff",
        "card_bg_elevated": "#f1f5f9",
        "card_border": "#e2e8f0",
        "card_border_highlight": "#0284c7",
        "text_primary": "#0f172a",
        "text_secondary": "#334155",
        "text_muted": "#64748b",
        "accent_blue": "#0284c7",
        "accent_indigo": "#4f46e5",
        "accent_purple": "#9333ea",
        "accent_cyan": "#0284c7",
        "accent_emerald": "#059669",
        "accent_amber": "#d97706",
        "accent_rose": "#dc2626",
        "rack_frame": "#f1f5f9",
        "blade_bg": "#ffffff",
        "plotly_template": "plotly_white",
        "plotly_paper": "rgba(0, 0, 0, 0)",
        "plotly_plot": "rgba(248, 250, 252, 0.85)",
        "plotly_grid": "rgba(226, 232, 240, 0.9)",
        "plotly_text": "#1e293b",
        "shadow": "0 2px 10px rgba(0, 0, 0, 0.05)",
        "badge_border": "rgba(0, 0, 0, 0.1)"
    }
}


def get_css(theme: str = "dark", nav_counter: int = 0) -> str:
    """Returns CSS corresponding to the current theme."""
    t = THEME_CONFIG.get(theme, THEME_CONFIG["dark"])
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
        
        :root {{
            --bg-primary: {t["bg_primary"]};
            --card-bg: {t["card_bg"]};
            --card-border: {t["card_border"]};
            --text-primary: {t["text_primary"]};
            --text-secondary: {t["text_secondary"]};
            --text-muted: {t["text_muted"]};
            --accent-blue: {t["accent_blue"]};
            --text-color: {t["text_primary"]};
            --background-color: {t["bg_primary"]};
            --secondary-background-color: {t["card_bg_elevated"]};
            --primary-color: {t["accent_blue"]};
        }}
        
        html, body, [class*="css"], .stApp, section.main, .main {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {t["bg_primary"]} !important;
            color: {t["text_primary"]} !important;
        }}
        
        /* Eliminate default Streamlit header gap & decoration */
        .block-container {{
            padding-top: 0.6rem !important;
            padding-bottom: 1.4rem !important;
            padding-left: 1.4rem !important;
            padding-right: 1.4rem !important;
            max-width: 1560px !important;
            background-color: transparent !important;
        }}
        
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0px !important;
            display: none !important;
        }}
        
        /* Hide all Streamlit default chrome & branding */
        .stDeployButton,
        [data-testid="stStatusWidget"],
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {{
            display: none !important;
            visibility: hidden !important;
        }}
        
        /* Base Container & Card System */
        .hw-card,
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
            background: {t["card_bg"]} !important;
            border: 1px solid {t["card_border"]} !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
            box-shadow: {t["shadow"]} !important;
            transition: border-color 0.2s ease !important;
        }}
        
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: none !important;
            box-shadow: none !important;
        }}
        
        .hw-card:hover, 
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child:hover {{
            border-color: {t["card_border_highlight"]} !important;
        }}
        
        .hw-card-title {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: -0.1px;
            color: {t["text_primary"]};
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
        }}

        /* Buttons & Navigation System */
        .stButton>button {{
            white-space: nowrap !important;
            border-radius: 7px !important;
            font-weight: 600 !important;
            font-size: 12px !important;
            letter-spacing: -0.1px !important;
            transition: all 0.18s ease !important;
        }}
        
        .stButton>button[kind="primary"] {{
            background: {t["accent_blue"]} !important;
            border: 1px solid {t["accent_blue"]} !important;
            color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        }}

        .stButton>button[kind="primary"]:hover {{
            background: #0284c7 !important;
            border-color: #0284c7 !important;
            transform: translateY(-1px) !important;
        }}
        
        .stButton>button[kind="secondary"] {{
            background: {t["card_bg"]} !important;
            border: 1px solid {t["card_border"]} !important;
            color: {t["text_primary"]} !important;
            box-shadow: none !important;
        }}

        .stButton>button[kind="secondary"]:hover {{
            border-color: {t["card_border_highlight"]} !important;
            background: {t["card_bg_elevated"]} !important;
            color: {t["text_primary"]} !important;
            transform: translateY(-1px) !important;
        }}

        /* Navigation Header Buttons */
        div[class*="st-key-nav_"] {{
            display: flex;
            justify-content: center;
        }}
        
        div[class*="st-key-nav_"] button {{
            white-space: nowrap !important;
            border-radius: 7px !important;
            height: 32px !important;
            min-height: 32px !important;
            padding: 0 10px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: -0.2px !important;
            transition: all 0.18s ease !important;
        }}

        div[class*="st-key-nav_"] button[kind="secondary"] {{
            background: {"rgba(255, 255, 255, 0.04)" if theme == "dark" else "rgba(255, 255, 255, 0.9)"} !important;
            border: 1px solid {"rgba(255, 255, 255, 0.08)" if theme == "dark" else "#e2e8f0"} !important;
            color: {"#94a3b8" if theme == "dark" else "#475569"} !important;
        }}

        div[class*="st-key-nav_"] button[kind="secondary"]:hover {{
            background: {"rgba(255, 255, 255, 0.08)" if theme == "dark" else "#ffffff"} !important;
            border-color: {"rgba(255, 255, 255, 0.2)" if theme == "dark" else "#cbd5e1"} !important;
            color: {"#f8fafc" if theme == "dark" else "#0f172a"} !important;
            transform: translateY(-1px) !important;
        }}

        div[class*="st-key-nav_"] button[kind="primary"] {{
            background: {"rgba(56, 189, 248, 0.16)" if theme == "dark" else "rgba(2, 132, 199, 0.12)"} !important;
            border: 1px solid {"rgba(56, 189, 248, 0.6)" if theme == "dark" else "#0284c7"} !important;
            color: {"#38bdf8" if theme == "dark" else "#0284c7"} !important;
            font-weight: 700 !important;
            transform: translateY(0) !important;
        }}

        div[class*="st-key-theme_toggle_btn"] button {{
            height: 32px !important;
            min-height: 32px !important;
            border-radius: 7px !important;
            font-size: 11.5px !important;
            font-weight: 600 !important;
            padding: 0 10px !important;
            background: {"rgba(255, 255, 255, 0.04)" if theme == "dark" else "rgba(255, 255, 255, 0.9)"} !important;
            border: 1px solid {t["card_border"]} !important;
            color: {t["text_secondary"]} !important;
        }}

        /* Workload Intensity Slider & SelectSlider: Force HeatWise Cyan #38bdf8 (replaces default red) */
        div[role="slider"] {{
            background-color: #38bdf8 !important;
            border-color: #38bdf8 !important;
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.45) !important;
        }}
        div[data-testid="stSliderThumbValue"] * {{
            color: #38bdf8 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 700 !important;
        }}
        div:has(> div[role="slider"]) + div {{
            background: #38bdf8 !important;
            background-color: #38bdf8 !important;
        }}
        [data-baseweb="slider"] div[role="slider"] {{
            background-color: #38bdf8 !important;
        }}
        [data-baseweb="slider"] [style*="background-color: rgb(255, 75, 75)"],
        [data-baseweb="slider"] [style*="background-color:#ff4b4b"] {{
            background-color: #38bdf8 !important;
        }}

        /* ==============================================================================
           LANDING PAGE CONTROL PLANE COMPONENTS
           ============================================================================== */
        
        /* Compact Cockpit Header Ribbon */
        .hw-cockpit-ribbon {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: {t["card_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 8px;
            padding: 6px 12px;
            margin-bottom: 8px;
            font-family: 'JetBrains Mono', monospace;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .hw-cockpit-left {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .hw-cockpit-status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: {t["accent_blue"]};
            box-shadow: 0 0 6px {t["accent_blue"]};
            flex-shrink: 0;
        }}

        .hw-cockpit-title {{
            font-size: 11px;
            font-weight: 800;
            color: {t["text_primary"]};
            letter-spacing: 0.8px;
        }}

        .hw-cockpit-sub {{
            font-size: 9.5px;
            color: {t["text_muted"]};
            letter-spacing: 0.4px;
            border-left: 1px solid {t["card_border"]};
            padding-left: 8px;
        }}

        .hw-cockpit-telemetry {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 10.5px;
            color: {t["text_secondary"]};
            flex-wrap: wrap;
        }}

        .hw-telem-item {{
            display: inline-flex;
            align-items: baseline;
            gap: 3px;
        }}

        .hw-telem-item b {{
            color: {t["text_primary"]};
            font-weight: 800;
        }}

        .hw-telem-item small {{
            color: {t["text_muted"]};
            font-size: 8.5px;
            font-weight: normal;
        }}

        .hw-telem-sep {{
            color: {t["card_border"]};
            font-size: 9.5px;
        }}

        /* Central Control Plane 3x4 Rack Grid */
        .hw-cp-container {{
            background: {t["card_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}

        .hw-crah-banner {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            padding: 3px 8px;
            background: {"rgba(56, 189, 248, 0.06)" if theme == "dark" else "rgba(2, 132, 199, 0.05)"};
            border: 1px dashed {"rgba(56, 189, 248, 0.3)" if theme == "dark" else "rgba(2, 132, 199, 0.3)"};
            border-radius: 5px;
            margin-bottom: 6px;
            font-size: 9.5px;
            font-weight: 700;
            color: {t["accent_blue"]};
            letter-spacing: 0.7px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }}

        .hw-crah-arrow {{
            font-size: 8.5px;
            opacity: 0.8;
        }}

        .hw-cp-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 7px;
            margin-bottom: 6px;
        }}

        .hw-cp-rack {{
            background: {t["blade_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 7px;
            padding: 6px 7px 5px 7px;
            position: relative;
            transition: all 0.18s ease;
        }}

        .hw-cp-rack:hover {{
            border-color: {t["accent_blue"]};
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
        }}

        .hw-cp-recommended {{
            border: 1.5px solid {t["accent_blue"]} !important;
            background: {"rgba(56, 189, 248, 0.08)" if theme == "dark" else "rgba(2, 132, 199, 0.06)"} !important;
            box-shadow: {"0 0 16px rgba(56, 189, 248, 0.35)" if theme == "dark" else "0 0 12px rgba(2, 132, 199, 0.2)"} !important;
        }}

        .hw-cp-target-badge {{
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            background: #0284c7;
            color: #ffffff;
            font-size: 8px;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            white-space: nowrap;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            z-index: 2;
        }}

        .hw-cp-rack-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }}

        .hw-cp-rack-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            font-weight: 800;
            color: {t["text_primary"]};
        }}

        .hw-cp-rack-temp {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            font-weight: 800;
        }}

        .hw-cp-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 8px;
            font-weight: 700;
            padding: 1px 4px;
            border-radius: 3px;
            letter-spacing: 0.3px;
        }}

        .hw-cp-zone-tag {{
            font-size: 8px;
            font-weight: 700;
            padding: 1px 4px;
            border-radius: 3px;
            font-family: 'JetBrains Mono', monospace;
        }}

        .hw-cp-blades-strip {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 2px;
            margin-bottom: 3px;
        }}

        .hw-cp-blade {{
            height: 4px;
            background: {"rgba(255, 255, 255, 0.06)" if theme == "dark" else "#e2e8f0"};
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .hw-cp-blade-led {{
            width: 3px;
            height: 3px;
            border-radius: 50%;
        }}

        .hw-cp-meter-row {{
            display: flex;
            justify-content: space-between;
            font-size: 8.5px;
            font-family: 'JetBrains Mono', monospace;
            color: {t["text_muted"]};
            margin-bottom: 1px;
        }}

        .hw-cp-meter-label {{
            font-weight: 600;
        }}

        .hw-cp-meter-val {{
            font-weight: 700;
            color: {t["text_primary"]};
        }}

        .hw-cp-bar-bg {{
            height: 3px;
            background: {"rgba(255, 255, 255, 0.06)" if theme == "dark" else "#e2e8f0"};
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 2px;
        }}

        .hw-cp-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}

        .hw-cp-sub-stats {{
            display: flex;
            justify-content: space-between;
            font-size: 8.5px;
            font-family: 'JetBrains Mono', monospace;
            color: {t["text_muted"]};
            margin-top: 3px;
            padding-top: 2px;
            border-top: 1px solid {"rgba(255, 255, 255, 0.05)" if theme == "dark" else "#e2e8f0"};
        }}

        .hw-cp-legend-strip {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 4px 8px;
            background: {"rgba(0, 0, 0, 0.2)" if theme == "dark" else "rgba(0, 0, 0, 0.03)"};
            border-radius: 5px;
            font-size: 9.5px;
            font-family: 'JetBrains Mono', monospace;
            color: {t["text_muted"]};
        }}

        .hw-cp-legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 600;
        }}

        .hw-cp-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}

        /* Right-Side Decision Panel */
        .hw-dec-container {{
            background: {t["card_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}

        .hw-dec-target-box {{
            background: {"rgba(56, 189, 248, 0.06)" if theme == "dark" else "rgba(2, 132, 199, 0.05)"};
            border: 1px solid {"rgba(56, 189, 248, 0.35)" if theme == "dark" else "rgba(2, 132, 199, 0.35)"};
            border-radius: 7px;
            padding: 7px 10px;
            margin-bottom: 6px;
        }}

        .hw-dec-target-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }}

        .hw-dec-target-id {{
            font-size: 21px;
            font-weight: 900;
            color: #38bdf8;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }}

        .hw-dec-target-badge {{
            font-size: 8.5px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            padding: 2px 7px;
            border-radius: 4px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #10b981;
            font-family: 'JetBrains Mono', monospace;
        }}

        .hw-dec-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 5px;
            margin-bottom: 6px;
            font-family: 'JetBrains Mono', monospace;
        }}

        .hw-dec-metric-box {{
            background: {t["blade_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 5px;
            padding: 4px 7px;
        }}

        .hw-dec-metric-label {{
            font-size: 8.5px;
            color: {t["text_muted"]};
            margin-bottom: 1px;
            text-transform: uppercase;
        }}

        .hw-dec-metric-val {{
            font-size: 13.5px;
            font-weight: 800;
            color: {t["text_primary"]};
        }}

        .hw-dec-rationale {{
            background: {"rgba(0, 0, 0, 0.25)" if theme == "dark" else "rgba(0, 0, 0, 0.04)"};
            border-left: 3px solid {t["accent_blue"]};
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 10.5px;
            color: {t["text_secondary"]};
            line-height: 1.35;
            margin-bottom: 6px;
        }}

        /* Architecture Specification Sheet (replaces SaaS cards) */
        .hw-arch-sheet {{
            background: {t["blade_bg"]};
            border: 1px solid {t["card_border"]};
            border-radius: 8px;
            padding: 10px 14px;
        }}

        .hw-arch-row {{
            display: flex;
            gap: 10px;
            margin-bottom: 7px;
            align-items: flex-start;
        }}

        .hw-arch-row:last-child {{
            margin-bottom: 0;
        }}

        .hw-arch-num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 800;
            color: {t["accent_blue"]};
            white-space: nowrap;
            margin-top: 1px;
        }}

        .hw-arch-title {{
            font-size: 11.5px;
            font-weight: 700;
            color: {t["text_primary"]};
            margin-bottom: 1px;
        }}

        .hw-arch-desc {{
            font-size: 10.5px;
            color: {t["text_secondary"]};
            line-height: 1.35;
        }}

        /* Benchmark Snapshot */
        .hw-bench-note {{
            font-size: 11px;
            color: {t["text_muted"]};
            line-height: 1.45;
            padding: 7px 10px;
            background: {"rgba(0, 0, 0, 0.2)" if theme == "dark" else "rgba(0, 0, 0, 0.03)"};
            border-radius: 5px;
            border-left: 3px solid #f59e0b;
            margin-top: 8px;
        }}

        /* Table styling */
        div[data-testid="stDataFrame"] {{
            border-radius: 8px !important;
            overflow: hidden !important;
            border: 1px solid {t["card_border"]} !important;
        }}

        /* Responsive */
        @media (max-width: 1200px) {{
            .hw-infra-strip {{ grid-template-columns: repeat(3, 1fr); }}
            .hw-pillars-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .hw-pipeline-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .hw-cp-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
    """


def get_thermal_color(temp: float) -> str:
    """Returns semantic hex color matching rack temperature."""
    if temp >= TEMP_MAX_SAFE:
        return "#dc2626"  # SLA Breach (>80°C)
    elif temp >= TEMP_CRITICAL:
        return "#ef4444"  # Hotspot (75-80°C)
    elif temp >= 65.0:
        return "#f59e0b"  # Warm (65-75°C)
    elif temp < 48.0:
        return "#10b981"  # Optimal Cool (<48°C)
    else:
        return "#38bdf8"  # Nominal (48-65°C)


def get_thermal_badge(temp: float) -> Tuple[str, str, str, str]:
    """Returns (status_text, text_color, badge_bg, badge_border)."""
    if temp >= TEMP_MAX_SAFE:
        return "SLA BREACH", "#dc2626", "rgba(220, 38, 38, 0.2)", "rgba(220, 38, 38, 0.5)"
    elif temp >= TEMP_CRITICAL:
        return "HOTSPOT", "#ef4444", "rgba(239, 68, 68, 0.2)", "rgba(239, 68, 68, 0.5)"
    elif temp >= 65.0:
        return "WARM", "#f59e0b", "rgba(245, 158, 11, 0.2)", "rgba(245, 158, 11, 0.4)"
    elif temp < 48.0:
        return "OPTIMAL", "#10b981", "rgba(16, 185, 129, 0.2)", "rgba(16, 185, 129, 0.4)"
    else:
        return "NOMINAL", "#38bdf8", "rgba(56, 189, 248, 0.15)", "rgba(56, 189, 248, 0.35)"


def render_control_plane_rack_grid(
    racks: Dict[int, Dict[str, Any]], 
    recommended_rack_id: Optional[int] = None, 
    selected_rack_id: Optional[int] = None, 
    theme: str = "dark"
) -> str:
    """
    Renders the central 3x4 facility rack topology grid as a miniature
    hyperscale control-plane telemetry map.
    """
    t = THEME_CONFIG.get(theme, THEME_CONFIG["dark"])
    
    html_parts = []
    html_parts.append('<div class="hw-cp-container">')
    html_parts.append('  <div class="hw-crah-banner">')
    html_parts.append('    <span class="hw-crah-arrow">▼</span>')
    html_parts.append('    <span>COLD AISLE SUPPLY AIRFLOW &nbsp;•&nbsp; CRAH INLET T<sub>amb</sub> = 22.0°C</span>')
    html_parts.append('    <span class="hw-crah-arrow">▼</span>')
    html_parts.append('  </div>')
    html_parts.append('  <div class="hw-cp-grid">')
    
    for r_id in range(1, 13):
        rack = racks.get(r_id, {})
        temp = rack.get('temperature', 50.0)
        gpu = rack.get('avg_gpu_util', 50.0)
        power_kw = rack.get('total_power_w', 7000.0) / 1000.0
        cooling = rack.get('cooling_capacity_pct', 90.0)
        headroom = rack.get('thermal_headroom', max(0.0, TEMP_CRITICAL - temp))
        
        status_text, temp_col, badge_bg, badge_border = get_thermal_badge(temp)
        is_rec = (recommended_rack_id is not None and r_id == recommended_rack_id)
        card_class = "hw-cp-rack" + (" hw-cp-recommended" if is_rec else "")
        
        special_tag = ""
        if r_id == 7:
            special_tag = '<span class="hw-cp-zone-tag" style="color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.12);">Ri=1.38 Hot</span>'
        elif r_id == 3:
            special_tag = '<span class="hw-cp-zone-tag" style="color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.12);">Ci=96% Cool</span>'
            
        target_badge = ""
        if is_rec:
            target_badge = '<div class="hw-cp-target-badge">★ RECOMMENDED TARGET</div>'
            
        blades_html = []
        for s in range(1, 5):
            blades_html.append(f'<div class="hw-cp-blade"><span class="hw-cp-blade-led" style="background:{temp_col}; box-shadow: 0 0 3px {temp_col};"></span></div>')
        blades_str = ''.join(blades_html)
        
        gpu_bar_col = '#ef4444' if gpu > 85 else ('#f59e0b' if gpu > 70 else ('#38bdf8' if theme == 'dark' else '#0284c7'))
        hr_bar_col = '#ef4444' if headroom < 4 else ('#f59e0b' if headroom < 10 else '#10b981')
        
        html_parts.append(f'''
        <div class="{card_class}">
          {target_badge}
          <div class="hw-cp-rack-head">
            <div style="display:flex; align-items:center; gap:5px;">
              <span class="hw-cp-rack-id">R{r_id:02d}</span>
              {special_tag}
            </div>
            <div style="display:flex; align-items:center; gap:5px;">
              <span class="hw-cp-rack-temp" style="color:{temp_col};">{temp:.1f}°C</span>
              <span class="hw-cp-badge" style="background:{badge_bg}; border: 1px solid {badge_border}; color:{temp_col};">{status_text}</span>
            </div>
          </div>
          <div class="hw-cp-blades-strip">{blades_str}</div>
          <div class="hw-cp-telemetry">
            <div class="hw-cp-meter-row">
              <span class="hw-cp-meter-label">GPU UTIL</span>
              <span class="hw-cp-meter-val">{gpu:.0f}%</span>
            </div>
            <div class="hw-cp-bar-bg">
              <div class="hw-cp-bar-fill" style="width: {min(100.0, gpu)}%; background: {gpu_bar_col};"></div>
            </div>
            <div class="hw-cp-meter-row" style="margin-top: 5px;">
              <span class="hw-cp-meter-label">HEADROOM</span>
              <span class="hw-cp-meter-val" style="color:{hr_bar_col};">+{headroom:.1f}°C</span>
            </div>
            <div class="hw-cp-bar-bg">
              <div class="hw-cp-bar-fill" style="width: {min(100.0, max(0.0, headroom/30.0*100))}%; background: {hr_bar_col};"></div>
            </div>
            <div class="hw-cp-sub-stats">
              <span>⚡ {power_kw:.1f} kW</span>
              <span>❄️ {cooling:.0f}% CRAH</span>
            </div>
          </div>
        </div>
        ''')
        
    html_parts.append('  </div>')
    html_parts.append('''
      <div class="hw-cp-legend-strip">
        <span class="hw-cp-legend-item"><span class="hw-cp-dot" style="background:#10b981;"></span>&lt;48°C Optimal</span>
        <span class="hw-cp-legend-item"><span class="hw-cp-dot" style="background:#38bdf8;"></span>48–65°C Nominal</span>
        <span class="hw-cp-legend-item"><span class="hw-cp-dot" style="background:#f59e0b;"></span>65–75°C Warm</span>
        <span class="hw-cp-legend-item"><span class="hw-cp-dot" style="background:#ef4444;"></span>≥75°C Hotspot (TEMP_CRITICAL)</span>
        <span class="hw-cp-legend-item"><span class="hw-cp-dot" style="background:#dc2626;"></span>≥80°C SLA Breach (TEMP_MAX_SAFE)</span>
      </div>
    ''')
    html_parts.append('</div>')
    return "\n".join(line.strip() for line in html_parts if line.strip())


def render_datacenter_rack_grid(racks: Dict[int, Dict[str, Any]], theme: str = "dark") -> str:
    """Compatibility wrapper that delegates to render_control_plane_rack_grid."""
    return render_control_plane_rack_grid(racks, theme=theme)


def create_facility_thermal_profile_chart(racks: Dict[int, Dict[str, Any]], theme: str = "dark") -> go.Figure:
    """Creates a 12-rack horizontal thermal profile bar chart with 75°C and 80°C threshold lines."""
    t = THEME_CONFIG.get(theme, THEME_CONFIG["dark"])
    
    y_labels = [f"R{i:02d}" for i in range(12, 0, -1)]
    temps = [racks[i]["temperature"] for i in range(12, 0, -1)]
    colors = [get_thermal_color(temp) for temp in temps]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_labels,
        x=temps,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{temp:.1f}°C" for temp in temps],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono, monospace", color=t["plotly_text"]),
        cliponaxis=False,
        hoverinfo="y+x"
    ))
    
    fig.add_vline(
        x=TEMP_CRITICAL,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="75°C Critical",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=9, family="JetBrains Mono, monospace")
    )
    fig.add_vline(
        x=TEMP_MAX_SAFE,
        line_dash="dot",
        line_color="#dc2626",
        line_width=1.5,
        annotation_text="80°C Max Safe",
        annotation_position="bottom right",
        annotation_font=dict(color="#dc2626", size=9, family="JetBrains Mono, monospace")
    )
    
    fig.update_layout(
        title=dict(text="<b>Facility Rack Thermal Profile (°C)</b>", font=dict(size=13, family="Inter, sans-serif")),
        xaxis=dict(range=[20, 96], title="Equilibrium Temperature (°C)", dtick=10),
        yaxis=dict(title="Rack ID", tickfont=dict(size=10, family="JetBrains Mono, monospace")),
        margin=dict(l=40, r=45, t=35, b=30),
        height=320
    )
    return apply_plotly_theme(fig, theme=theme, height=320)


def apply_plotly_theme(fig: go.Figure, theme: str = "dark", height: int = 340) -> go.Figure:
    """Applies consistent light or dark theme to any Plotly figure."""
    t = THEME_CONFIG.get(theme, THEME_CONFIG["dark"])
    
    fig.update_layout(
        template=t["plotly_template"],
        paper_bgcolor=t["plotly_paper"],
        plot_bgcolor=t["plotly_plot"],
        font=dict(family="Inter, sans-serif", color=t["plotly_text"], size=12),
        margin=dict(l=35, r=20, t=35, b=35),
        height=height,
        hoverlabel=dict(
            bgcolor=t["card_bg"],
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    fig.update_xaxes(
        gridcolor=t["plotly_grid"],
        linecolor=t["plotly_grid"],
        zerolinecolor=t["plotly_grid"],
        tickfont=dict(color=t["plotly_text"], size=10),
        title_font=dict(color=t["plotly_text"], size=11)
    )
    fig.update_yaxes(
        gridcolor=t["plotly_grid"],
        linecolor=t["plotly_grid"],
        zerolinecolor=t["plotly_grid"],
        tickfont=dict(color=t["plotly_text"], size=10),
        title_font=dict(color=t["plotly_text"], size=11)
    )
    return fig
