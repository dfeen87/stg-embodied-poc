# stg-embodied-poc

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MuJoCo 3.x](https://img.shields.io/badge/MuJoCo-3.x-green)
![License Non-Commercial](https://img.shields.io/badge/license-Non--Commercial-lightgrey)
[![CI](https://github.com/dfeen87/stg-embodied-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/dfeen87/stg-embodied-poc/actions/workflows/ci.yml)


**Minimal MuJoCo Proof-of-Concept for the Spiral-Time Governor (STG)**

---

## Purpose

The **Spiral-Time Governor** (STG) is a deterministic external supervision layer that wraps a black-box LLM and gates its outputs (claims + actions) based on a scalar instability functional ΔΦ(t), as described in §3.2 of the companion paper.  This repository demonstrates that the STG architecture transfers from the synthetic noise model evaluated in the paper to a real physics simulator, using dm_control's built-in quadruped domain ("escape" task) as a proxy for the custom ANYmal-class terrain tasks described in the paper.  All governor parameters, thresholds, and evaluation metrics are identical to the synthetic testbed v2.2.

> **Note:** This PoC uses the dm_control quadruped "escape" task as a physics-grounded proxy for the custom ANYmal-class terrain locomotion tasks described in the paper.  The escape task provides realistic contact dynamics and proprioceptive observations without requiring a licensed ANYmal model.

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 2. Run the full pipeline (all conditions, seeds 0–9 except 5)
bash scripts/run_all.sh

# 3. Run a quick smoke test (3 seeds, 2 conditions)
python run_experiment.py --seeds 0 1 2 --conditions baseline governor

# 4. Run with verbose per-step output
python run_experiment.py --conditions governor --seeds 0 --verbose

# 5. Run robustness check (held-out seeds 40–49)
bash scripts/run_robustness.sh

# 6. Recompute metrics from saved results
python analysis/compute_metrics.py --results_dir results/
```

Results are written to `results/` as CSV and JSON files (see [Output Files](#output-files) below).

---

## Full Reproducibility (One Command)

**Environment:** Python 3.10+, MuJoCo 3.x (see `requirements.txt` for full dependencies).

All random seeds are fixed in [`config.py`](config.py) (`DEFAULT_SEEDS = list(range(10))`, `BOOTSTRAP_SEED = 77`).

To regenerate **all figures** used in the paper:

```bash
python images/generate_plots.py
```

This script reads the embedded simulation data and writes all eight figures (`fig1_*.png` – `fig8_*.png`) to the `images/` directory.

---

## Directory Structure

```
stg-embodied-poc/
├── README.md
├── LICENSE                     # Non-Commercial
├── PAPER.md                    # Companion paper (manuscript)
├── VALIDATION_ANALYSIS.md      # Detailed validation results and statistical analysis
├── CITATION.cff                # Citation metadata for the companion paper
├── requirements.txt
├── config.py                   # Single source of truth for all parameters
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml              # CI pipeline (lint + tests, Python 3.10 & 3.11)
├── envs/
│   ├── __init__.py
│   └── quadruped_terrain.py    # dm_control quadruped wrapper + oracle
├── governor/
│   ├── __init__.py
│   └── spiral_time_governor.py # STG implementation (math from paper)
├── llm_mock/
│   ├── __init__.py
│   ├── mock_llm_agent.py       # Deterministic mock LLM agent
│   └── real_llm_agent.py       # Real LLM agent (API-backed)
├── analysis/
│   ├── __init__.py
│   └── compute_metrics.py      # Metric computation from episode logs
├── images/
│   ├── generate_plots.py       # Script to regenerate all figures
│   ├── fig1_violations_by_condition.png
│   ├── fig2_mode_distribution.png
│   ├── fig3_hallucination_rate.png
│   ├── fig4_signal_statistics.png
│   ├── fig5_per_seed_violations_heatmap.png
│   ├── fig6_success_rate.png
│   ├── fig7_performance_overhead.png
│   └── fig8_robustness_summary.png
├── run_experiment.py           # Main experiment runner (CLI)
├── run_replay.py               # CLI for the counterfactual replay system
├── run_sensitivity_sweep.py    # One-at-a-time sensitivity sweep over ΔΦ weights (α, β, γ, δ)
├── replay/
│   ├── __init__.py
│   └── counterfactual_replay.py  # Deterministic record-then-replay comparison (raw vs. STG-filtered)
├── scripts/
│   ├── run_all.sh              # Full pipeline (all conditions, seeds 0–9)
│   └── run_robustness.sh       # Robustness check (governor, seeds 40–49)
├── tests/
│   ├── test_governor.py
│   ├── test_env.py
│   ├── test_real_llm_agent.py
│   ├── test_replay.py
│   └── test_sensitivity_sweep.py
└── results/
    ├── .gitkeep
    ├── ablations.csv           # Per-condition ablation results
    ├── metrics.csv             # Aggregated metrics across seeds/conditions
    ├── sensitivity_delta_phi.csv  # Sensitivity sweep results for ΔΦ weights
    └── replay/                 # Per-seed counterfactual replay JSON outputs
```

---

## Parameter Table

### STG Fixed Parameters (§3.2 of paper — defined once in `governor/spiral_time_governor.py`)

| Parameter | Value | Description | Paper §  |
|-----------|-------|-------------|----------|
| `wR`      | 0.30  | Weight for structure deviation ΔR in coherence score φ | §3.2 |
| `wI`      | 0.40  | Weight for information deviation ΔI in coherence score φ | §3.2 |
| `wC`      | 0.30  | Weight for coherence deviation ΔC in coherence score φ | §3.2 |
| `α`       | 0.25  | Weight for ΔR in instability functional ΔΦ | §3.2 |
| `β`       | 0.35  | Weight for ΔI in instability functional ΔΦ | §3.2 |
| `γ`       | 0.25  | Weight for ΔC in instability functional ΔΦ | §3.2 |
| `δ`       | 0.15  | Weight for torsion \|χ(t)\| in instability functional ΔΦ | §3.2 |
| `τ₁`      | 0.25  | EXECUTE → VERIFY threshold | §3.3 |
| `τ₂`      | 0.55  | VERIFY → SAFE threshold | §3.3 |
| `φ₀`      | 0.75  | Initial coherence score | §3.2 |

### Experimental Design

| Parameter            | Value                    | Description |
|----------------------|--------------------------|-------------|
| `MAX_STEPS`          | 120                      | Maximum steps per episode |
| `DEFAULT_SEEDS`      | `list(range(10))`        | Seeds 0–9 for main experiment |
| `BOOTSTRAP_RESAMPLES`| 2000                     | Bootstrap resamples for CIs |
| `BOOTSTRAP_SEED`     | 77                       | Seed for bootstrap RNG |
| `SIGNIFICANCE_LEVEL` | 0.05                     | α-level for hypothesis tests |

### Experimental Conditions

| Condition    | Ablation         | Hallucination Prob | Description |
|--------------|------------------|--------------------|-------------|
| `baseline`   | `always_execute` | 0.45               | No governor; all LLM actions executed |
| `governor`   | `none`           | 0.45               | Full STG with torsion term |
| `ablation_a` | `no_delta`       | 0.45               | STG with δ=0 (torsion disabled) |
| `rag`        | `always_execute` | 0.30               | No governor; reduced hallucination (RAG proxy) |

### CLI Arguments (`run_experiment.py`)

| Argument       | Default           | Description |
|----------------|-------------------|-------------|
| `--seeds`      | `[0, 1, 2, 3, 4]` | Random seeds |
| `--conditions` | `all`             | Conditions to run |
| `--task`       | `terrain`         | Task name (reserved for future multi-task) |
| `--n_claims`   | `3`               | LLM claims per step |
| `--output_dir` | `results/`        | Output directory |
| `--verbose`    | `False`           | Verbose per-step logging |
| `--use-real-llm` | `False`         | Use `RealLLMAgent` (real OpenAI API) instead of `MockLLMAgent`; requires `OPENAI_API_KEY` |

---

## Output Files

After running `run_experiment.py`, the `results/` directory contains:

| File | Description |
|------|-------------|
| `results.json` | Flat per-episode metrics (H_T, violations, success, n_steps) for all conditions/seeds |
| `summary.csv`  | One row per episode with all computed metrics (written by `compute_metrics.py`) |
| `metrics.csv`  | Per-condition H_T, violation_rate and success_rate (three primary metrics) |
| `ablations.csv` | Per-episode ablation comparison (governor vs δ=0, remove_I, remove_C) |
| `<condition>_steps.csv` | Per-step logs for each condition (phi, delta_phi, mode, reward, oracle, …) |

---

## Limitations

### Practical / Engineering Limitations

1. **Synthetic vs. physics:** The mock LLM agent produces deterministic sinusoidal actions and rule-based claims.  Real LLM integration would require an API call layer (not included in this PoC).
2. **Oracle simplifications:** The `oracle()` method approximates terrain class from the torso x-position and contact count from geom-name matching.  A production system would use richer sensor fusion.
3. **Task proxy:** The dm_control quadruped "escape" task is a physics-grounded but simplified proxy for the ANYmal-class terrain tasks in the paper.  Custom terrain assets, gait controllers, and task reward shaping are outside the scope of this PoC.
4. **Single-process execution:** Episodes are run sequentially; wall-clock time scales linearly with `n_seeds × n_conditions × MAX_STEPS`.

### Scientific / Methodological Limitations

The following limitations are disclosed in the interest of academic transparency and to support appropriate interpretation of the reported results.

- **Mock LLM vs. Real Reasoning.**  The deterministic mock LLM in `llm_mock/mock_llm_agent.py` does not represent real reasoning or real‑world LLM variability.  The mock agent is used deliberately to isolate the effect of the Spiral Time Governor (STG) without introducing confounding factors such as stochastic token sampling or prompt sensitivity.  A real LLM condition is included only to demonstrate robustness to non‑deterministic outputs; it does not constitute a claim of cognitive modeling.

- **Simplified Environment (dm_control Quadruped).**  `envs/quadruped_terrain.py` uses a controlled dm_control locomotion task as a proxy for embodied decision‑making.  This environment does not capture full real‑world terrain complexity, multi‑contact dynamics, or hardware constraints such as actuator saturation, communication latency, or mechanical compliance.  Results should therefore be interpreted as simulation‑level evidence and should not be taken as direct evidence of transfer to physical robotic platforms.

- **Oracle Simplifications.**  The oracle used for action verification is simplified and relies on idealized state information extracted directly from the simulator.  It does not model full sensor uncertainty, partial observability, or real‑world perception errors.  Noise‑augmented oracle experiments are included to partially address this gap; however, they remain approximations and do not replace a fully realistic perception pipeline.

- **Scope of the STG.**  The Spiral Time Governor is a safety filter, not a hallucination‑reduction method.  STG does not modify the LLM's internal reasoning process, alter its weights, or reduce the rate at which the model produces hallucinated outputs.  Its sole function is to prevent unsafe actions from being executed; hallucinated actions that pass the safety threshold are still executed.

- **Limited Task Diversity.**  The current proof‑of‑concept focuses on a single locomotion task (dm_control quadruped "escape") and a limited set of action types.  Broader task diversity—spanning manipulation, navigation, multi‑agent coordination, and varied reward structures—is required before claims can be generalised beyond the reported setting.

- **Computational Constraints.**  Multi‑seed experiments (e.g., seeds 0–9 or 40–49) are limited by the available compute budget.  The reported confidence intervals and effect sizes are derived from these seed sets; larger parameter sweeps and additional held‑out seeds would further strengthen statistical confidence in the reported findings.

- **Reproducibility Notes.**  Results may vary slightly across hardware configurations, dm_control versions, and LLM API responses.  Minor numerical differences are expected due to floating‑point non‑determinism, platform‑specific physics solver behaviour, and non‑deterministic LLM sampling.  Core qualitative findings are expected to be stable across these variations, but exact numerical values should not be treated as perfectly reproducible without fixing all environmental dependencies.

---

## Critique

The following observations highlight areas for improvement that go beyond the limitations already disclosed above.

### Code / Architecture

- **Duplicated constraint checker.** The `_constraint_checker` function (‖action‖₂ < 8.0) is copy-pasted verbatim in both `run_experiment.py` and `replay/counterfactual_replay.py`.  It should live once in a shared module (e.g., `envs/` or a `utils.py`) and be imported in both places.

- **Unused return value in `_record_phase`.** `_record_phase` returns `initial_obs`, but neither `_replay_track_a` nor `_replay_track_b` uses it — both call `env.reset()` independently.  The dead return value is misleading and should be removed.

- **No `setup.py` / `pyproject.toml`.** The project is not installable as a package, which forces every import to rely on the working directory being the repo root.  Adding a minimal `pyproject.toml` would make dependency management and CI more robust.

- **Single-process execution is not parallelised.** The README documents that "episodes are run sequentially; wall-clock time scales linearly."  Adding optional `multiprocessing` support (e.g., `--n_workers`) would make the sensitivity sweep and robustness checks significantly faster at no algorithmic cost.

- **Hard-coded output filenames in `run_replay.py`.** The aggregate JSON is always written as `replay_aggregate_{condition}.json`; running the same condition twice silently overwrites it.  Using a timestamp suffix or `--force-overwrite` flag would be safer.

### Repository Hygiene

- **Committed result CSVs.** `results/ablations.csv`, `results/metrics.csv`, and `results/sensitivity_delta_phi.csv` are tracked by git.  For a reproducibility repo the canonical pattern is to `.gitignore` all generated files and let the reader regenerate them via the documented commands.  Committing results conflates source and artefact, and risks stale data if the code changes.

- **Wrong repo name in README H1.** The heading previously read `stg-mujoco-poc` (the earlier working title) instead of `stg-embodied-poc` (the canonical repository name).  Fixed in this update.

- **`results/replay/` not in the original tree.** The directory (and its `.gitkeep`) existed on disk but was absent from the documented tree.  Fixed in this update.

- **`replay/` module and `run_replay.py` not in the original tree.** Both were present in the codebase but missing from the Directory Structure section.  Fixed in this update.

### Testing

- **No integration test for `run_replay.py`.** `test_replay.py` tests the library function `run_counterfactual_replay` directly, but the CLI entry-point (`main()`) has no automated coverage.  A smoke-test that exercises the CLI with `--seeds 0 --condition governor` would close this gap.

- **CI does not run the sensitivity sweep test by default.**  `tests/test_sensitivity_sweep.py` exists but `ci.yml` should be inspected to confirm it is included in the test matrix; sweep tests can be slow and are sometimes accidentally excluded.

---


If you use this code, please cite the companion paper (placeholder — update when published):

```bibtex
@article{stg2026,
  title   = {Deterministic Spiral-Time Governance for Hallucination Suppression in LLM-Controlled Climbing and Walking Robots},
  author  = {Krüger, Marcel and Feeney, Don Michael Jr.},
  journal = {Journal of Climbing and Walking Robots},
  year    = {2026},
}
```
---

Non-Commercial License - see [`LICENSE`](LICENSE) file.

---

## Acknowledgments

This repository was developed with a combination of original ideas, hands‑on coding, and support from advanced AI systems. I would like to acknowledge **Microsoft Copilot** and **Anthropic Claude**, and for their meaningful assistance in refining concepts, improving clarity, and strengthening the overall quality of this code. 

