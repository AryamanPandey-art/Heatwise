# HeatWise Experimental Suite

This directory contains the isolated experimental research suite for the HeatWise project.

## Directory Structure

```
experiments/
├── __init__.py               <- Package initialization
├── dataset_utils.py          <- Sequential trajectory dataset generators with episode metadata
├── eval_utils.py             <- Standardized 250-decision dynamic closed-loop oracle harness
├── exp_runner.py             <- Master orchestration CLI
├── exp_a_scaling.py          <- Experiment A: Dataset scaling (3.5k vs 10k vs 20k)
├── exp_b_temporal.py         <- Experiment B: Episode-grouped and temporal block validation
├── exp_c_holdout.py          <- Experiment C: Leave-one-scenario-out generalization
├── exp_d_extratrees.py       <- Experiment D: ExtraTreesRegressor evaluation
├── exp_e_histboost.py        <- Experiment E: HistGradientBoostingRegressor evaluation
├── uncertainty.py            <- Experiment F: Zero-dependency tree-variance uncertainty
├── ablations.py              <- Experiment G: Feature & heuristic ablation studies
├── extended_benchmarks.py    <- Experiments H, I, J, K: Multi-seed and stress benchmarks
├── latency_profiler.py       <- Phase 11: System & operation latency benchmarks
├── plot_utils.py             <- Publication plot generators
├── results/                  <- Machine-readable CSV and JSON artifacts
│   ├── FINAL_EXPERIMENT_REPORT.md
│   ├── dataset_scaling.csv
│   ├── temporal_validation.csv
│   ├── scenario_holdout.csv
│   ├── model_comparison.csv
│   ├── uncertainty.csv
│   ├── ablations.csv
│   ├── multiseed_benchmark.csv
│   ├── bursty_stress.csv
│   ├── concurrency_stress.csv
│   ├── sustained_load.csv
│   └── latency.csv
└── plots/                    <- Publication-grade PNG charts
```

## Running the Suite

Execute the entire suite sequentially:

```bash
python3 experiments/exp_runner.py
```

All production code outside `experiments/` remains strictly unmodified and protected.
