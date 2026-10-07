"""
HeatWise Publication Plotting Utilities
Generates publication-quality charts from experiment result artifacts.
Strictly plots measured values from experiment outputs; zero fabricated data.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots"))

# Publication style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3
})

COLORS = {
    "primary": "#3b82f6",    # Blue
    "accent": "#10b981",     # Emerald
    "danger": "#ef4444",     # Red
    "warning": "#f59e0b",    # Amber
    "purple": "#8b5cf6",     # Purple
    "slate": "#64748b",      # Slate
    "round_robin": "#6b7280",
    "least_loaded": "#f97316",
    "heatwise": "#10b981"
}


def generate_all_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n--- Generating Publication-Quality Plots ---")

    # Plot 1: Dataset Size vs Validation Error
    try:
        p1_path = os.path.join(RESULTS_DIR, "dataset_scaling.csv")
        if os.path.exists(p1_path):
            df = pd.read_csv(p1_path)
            fig, ax1 = plt.subplots(figsize=(6.5, 4))
            ax2 = ax1.twinx()

            x = [str(c) for c in df["sample_count"]]
            l1 = ax1.plot(x, df["static_mae"], marker="o", color=COLORS["primary"], lw=2, label="Static MAE (°C)")
            l2 = ax1.plot(x, df["dynamic_mae"], marker="s", color=COLORS["danger"], lw=2, label="Dynamic MAE (°C)")
            l3 = ax2.plot(x, df["dynamic_top1_pct"], marker="^", color=COLORS["accent"], lw=2, linestyle="--", label="Top-1 Agreement (%)")

            ax1.set_xlabel("Dataset Size (Samples)")
            ax1.set_ylabel("Mean Absolute Error (°C)")
            ax2.set_ylabel("Top-1 Agreement (%)")
            ax1.set_title("Experiment A: Telemetry Dataset Scaling vs. Generalization Error")

            lines = l1 + l2 + l3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="center right")

            out_file = os.path.join(PLOTS_DIR, "1_dataset_scaling.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 1: {e}")

    # Plot 2: Random Split vs Temporal vs Episode-Grouped
    try:
        p2_path = os.path.join(RESULTS_DIR, "temporal_validation.csv")
        if os.path.exists(p2_path):
            df = pd.read_csv(p2_path)
            fig, ax = plt.subplots(figsize=(7.5, 4))
            x_indices = np.arange(len(df))
            width = 0.35

            ax.bar(x_indices - width/2, df["static_mae"], width, label="Static Test MAE (°C)", color=COLORS["primary"])
            ax.bar(x_indices + width/2, df["dynamic_mae"], width, label="Dynamic Decision MAE (°C)", color=COLORS["warning"])

            ax.set_xticks(x_indices)
            ax.set_xticklabels([m.replace(" Split", "\nSplit") for m in df["split_method"]])
            ax.set_ylabel("Mean Absolute Error (°C)")
            ax.set_title("Experiment B: Data Leakage Comparison across Splitting Protocols")
            ax.legend()

            out_file = os.path.join(PLOTS_DIR, "2_temporal_split_comparison.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 2: {e}")

    # Plot 3: Scenario Holdout Performance
    try:
        p3_path = os.path.join(RESULTS_DIR, "scenario_holdout.csv")
        if os.path.exists(p3_path):
            df = pd.read_csv(p3_path)
            fig, ax = plt.subplots(figsize=(8, 4))
            x_indices = np.arange(len(df))

            ax.bar(x_indices, df["dynamic_mae"], color=COLORS["purple"], width=0.5, label="Held-Out Dynamic MAE (°C)")
            ax.set_xticks(x_indices)
            ax.set_xticklabels([s.replace(" ", "\n") for s in df["held_out_scenario"]])
            ax.set_ylabel("Dynamic Decision MAE (°C)")
            ax.set_title("Experiment C: Leave-One-Scenario-Out Generalization Error")

            for i, v in enumerate(df["dynamic_mae"]):
                ax.text(i, v + 0.1, f"{v:.2f}°C", ha="center", fontsize=9)

            out_file = os.path.join(PLOTS_DIR, "3_scenario_holdout.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 3: {e}")

    # Plot 4: Model Comparison (RF vs ExtraTrees vs HistGradientBoosting)
    try:
        p4_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
        if os.path.exists(p4_path):
            df = pd.read_csv(p4_path)
            fig, ax1 = plt.subplots(figsize=(7, 4))
            x_indices = np.arange(len(df))
            width = 0.35

            ax1.bar(x_indices - width/2, df["static_mae"], width, label="Static MAE (°C)", color=COLORS["primary"])
            ax1.bar(x_indices + width/2, df["dynamic_mae"], width, label="Dynamic MAE (°C)", color=COLORS["accent"])

            ax1.set_xticks(x_indices)
            ax1.set_xticklabels([m.split()[0] for m in df["model_name"]])
            ax1.set_ylabel("MAE (°C)")
            ax1.set_title("Experiment D & E: Model Architecture Comparison")
            ax1.legend()

            out_file = os.path.join(PLOTS_DIR, "4_model_comparison.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 4: {e}")

    # Plot 5: Uncertainty Distribution & Safety Boundary
    try:
        p5_path = os.path.join(RESULTS_DIR, "uncertainty.csv")
        if os.path.exists(p5_path):
            df = pd.read_csv(p5_path)
            fig, ax1 = plt.subplots(figsize=(6.5, 4))
            ax2 = ax1.twinx()

            x = [f"k={k}" for k in df["k_multiplier"]]
            l1 = ax1.plot(x, df["rejected_by_conservative_bound"], marker="o", color=COLORS["danger"], lw=2, label="Preemptive Rejections")
            l2 = ax2.plot(x, df["top1_agreement_pct"], marker="s", color=COLORS["primary"], lw=2, linestyle="--", label="Top-1 Agreement (%)")

            ax1.set_xlabel("Uncertainty Multiplier (T_bound = mu + k*sigma)")
            ax1.set_ylabel("Uncertainty-Triggered Rejections")
            ax2.set_ylabel("Top-1 Agreement (%)")
            ax1.set_title("Experiment F: Tree-Variance Risk Bound vs. Decision Conservatism")

            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper left")

            out_file = os.path.join(PLOTS_DIR, "5_uncertainty_safety_boundary.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 5: {e}")

    # Plot 6: Ablations
    try:
        p6_path = os.path.join(RESULTS_DIR, "ablations.csv")
        if os.path.exists(p6_path):
            df = pd.read_csv(p6_path)
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            y_pos = np.arange(len(df))

            labels = [v.replace("G", "").replace("_", " ") for v in df["variant_id"]]
            ax.barh(y_pos, df["dynamic_mae"], color=COLORS["slate"], height=0.55)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Dynamic Decision MAE (°C)")
            ax.set_title("Experiment G: Feature & Heuristic Ablation Contributions")

            for i, v in enumerate(df["dynamic_mae"]):
                ax.text(v + 0.05, i, f"{v:.2f}°C", va="center", fontsize=9)

            out_file = os.path.join(PLOTS_DIR, "6_ablation_comparison.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 6: {e}")

    # Plot 7: Multi-Seed Robustness (Peak Temp & Spatial Variance)
    try:
        p7_path = os.path.join(RESULTS_DIR, "multiseed_benchmark.csv")
        if os.path.exists(p7_path):
            df = pd.read_csv(p7_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Grouped metrics
            scheds = ["Round Robin", "Least Loaded", "HeatWise"]
            means_peak = [df[df["scheduler"] == s]["peak_temp"].mean() for s in scheds]
            stds_peak = [df[df["scheduler"] == s]["peak_temp"].std() for s in scheds]
            means_std = [df[df["scheduler"] == s]["temp_std"].mean() for s in scheds]
            stds_std = [df[df["scheduler"] == s]["temp_std"].std() for s in scheds]

            colors = [COLORS["round_robin"], COLORS["least_loaded"], COLORS["heatwise"]]

            ax1.bar(scheds, means_peak, yerr=stds_peak, capsize=5, color=colors, alpha=0.85)
            ax1.set_ylabel("Peak Temperature (°C)")
            ax1.set_title("Peak Temperature (Mean ± Std, 5 Seeds)")

            ax2.bar(scheds, means_std, yerr=stds_std, capsize=5, color=colors, alpha=0.85)
            ax2.set_ylabel("Thermal Standard Deviation (°C)")
            ax2.set_title("Thermal Variance (Mean ± Std, 5 Seeds)")

            out_file = os.path.join(PLOTS_DIR, "7_multiseed_thermal_profiles.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 7: {e}")

    # Plot 8: Multi-Seed Hotspots & SLA Violations
    try:
        p8_path = os.path.join(RESULTS_DIR, "multiseed_benchmark.csv")
        if os.path.exists(p8_path):
            df = pd.read_csv(p8_path)
            fig, ax = plt.subplots(figsize=(7, 4))
            scheds = ["Round Robin", "Least Loaded", "HeatWise"]
            x_indices = np.arange(len(scheds))
            width = 0.35

            means_hot = [df[df["scheduler"] == s]["hotspots"].mean() for s in scheds]
            stds_hot = [df[df["scheduler"] == s]["hotspots"].std() for s in scheds]
            means_sla = [df[df["scheduler"] == s]["sla_violations"].mean() for s in scheds]
            stds_sla = [df[df["scheduler"] == s]["sla_violations"].std() for s in scheds]

            ax.bar(x_indices - width/2, means_hot, width, yerr=stds_hot, capsize=4, label="Hotspots (≥75°C)", color=COLORS["warning"])
            ax.bar(x_indices + width/2, means_sla, width, yerr=stds_sla, capsize=4, label="SLA Breaches (≥80°C)", color=COLORS["danger"])

            ax.set_xticks(x_indices)
            ax.set_xticklabels(scheds)
            ax.set_ylabel("Incident Count")
            ax.set_title("Experiment H: Hotspot & SLA Incident Comparison Across 5 Seeds")
            ax.legend()

            out_file = os.path.join(PLOTS_DIR, "8_multiseed_incidents.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 8: {e}")

    # Plot 9: Stress Tests Summary
    try:
        stress_files = [
            ("Bursty Stress", os.path.join(RESULTS_DIR, "bursty_stress.csv")),
            ("Concurrency Stress", os.path.join(RESULTS_DIR, "concurrency_stress.csv")),
            ("Sustained High-Load", os.path.join(RESULTS_DIR, "sustained_load.csv"))
        ]
        stress_data = []
        for regime, fp in stress_files:
            if os.path.exists(fp):
                d = pd.read_csv(fp)
                for _, row in d.iterrows():
                    stress_data.append({
                        "regime": regime,
                        "scheduler": row["scheduler"],
                        "peak_temp": row["peak_temp"],
                        "sla_violations": row["sla_violations"]
                    })
        if stress_data:
            s_df = pd.DataFrame(stress_data)
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            regimes = s_df["regime"].unique()
            x = np.arange(len(regimes))
            width = 0.25

            rr_peaks = [s_df[(s_df["regime"] == r) & (s_df["scheduler"] == "Round Robin")]["peak_temp"].values[0] for r in regimes]
            ll_peaks = [s_df[(s_df["regime"] == r) & (s_df["scheduler"] == "Least Loaded")]["peak_temp"].values[0] for r in regimes]
            hw_peaks = [s_df[(s_df["regime"] == r) & (s_df["scheduler"] == "HeatWise")]["peak_temp"].values[0] for r in regimes]

            ax.bar(x - width, rr_peaks, width, label="Round Robin", color=COLORS["round_robin"])
            ax.bar(x, ll_peaks, width, label="Least Loaded", color=COLORS["least_loaded"])
            ax.bar(x + width, hw_peaks, width, label="HeatWise", color=COLORS["heatwise"])

            ax.set_xticks(x)
            ax.set_xticklabels(regimes)
            ax.set_ylabel("Peak Temperature (°C)")
            ax.set_title("Experiments I, J, K: Peak Temperatures Under Stress Regimes")
            ax.set_ylim(bottom=50)
            ax.legend()

            out_file = os.path.join(PLOTS_DIR, "9_stress_test_comparison.png")
            plt.savefig(out_file)
            plt.close()
            print(f"Generated {out_file}")
    except Exception as e:
        print(f"Failed Plot 9: {e}")

    print("--- Plot Generation Complete ---\n")


if __name__ == "__main__":
    generate_all_plots()
