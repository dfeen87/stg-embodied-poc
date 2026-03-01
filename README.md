# stg-mujoco-poc

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MuJoCo 3.x](https://img.shields.io/badge/MuJoCo-3.x-green)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

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

# 3. Run a quick smoke test (2 seeds, 2 conditions)
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

## Directory Structure

```
stg-embodied-poc/
├── README.md
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
│   └── mock_llm_agent.py       # Deterministic mock LLM agent
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
│   └── fig7_performance_overhead.png
├── run_experiment.py           # Main experiment runner (CLI)
├── scripts/
│   ├── run_all.sh              # Full pipeline (all conditions, seeds 0–9)
│   └── run_robustness.sh       # Robustness check (governor, seeds 40–49)
├── tests/
│   ├── test_governor.py
│   └── test_env.py
└── results/
    └── .gitkeep
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

---

## Output Files

After running `run_experiment.py`, the `results/` directory contains:

| File | Description |
|------|-------------|
| `results.json` | Flat per-episode metrics (H_T, violations, success, n_steps) for all conditions/seeds |
| `summary.csv`  | One row per episode with all computed metrics (written by `compute_metrics.py`) |
| `<condition>_steps.csv` | Per-step logs for each condition (phi, delta_phi, mode, reward, oracle, …) |

---

## Known Limitations

1. **Synthetic vs. physics:** The mock LLM agent produces deterministic sinusoidal actions and rule-based claims.  Real LLM integration would require an API call layer (not included in this PoC).
2. **Oracle simplifications:** The `oracle()` method approximates terrain class from the torso x-position and contact count from geom-name matching.  A production system would use richer sensor fusion.
3. **Task proxy:** The dm_control quadruped "escape" task is a physics-grounded but simplified proxy for the ANYmal-class terrain tasks in the paper.  Custom terrain assets, gait controllers, and task reward shaping are outside the scope of this PoC.
4. **Single-process execution:** Episodes are run sequentially; wall-clock time scales linearly with `n_seeds × n_conditions × MAX_STEPS`.

---

## Citation

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

## Acknowledgements

## Acknowledgments

This repository was developed with a combination of original ideas, hands‑on coding, and support from advanced AI systems. I would like to acknowledge **Microsoft Copilot**, **Anthropic Claude**, and **Google Jules** for their meaningful assistance in refining concepts, improving clarity, and strengthening the overall quality of this work. Original science and math was created by the authors. 

