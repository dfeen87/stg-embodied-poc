# Changelog

All notable changes to this project will be documented in this file.

---

## v1.1.0 — Oracle Correction & Paper Refinements

**Released:** 2026-02-28

### Overview

This release corrects a critical bug in the torso-uprightness oracle that affected all
embodied evaluation results reported in PAPER.md §8.2 and Table 5, and replaces fabricated
benchmark figures in `VALIDATION_ANALYSIS.md` with actual computed values.  All code and
statistical parameters remain identical to v1.0.0; only the oracle predicate and the
documents that depend on it have changed.

---

### Bug Fixes

#### Critical — `torso_upright` oracle (`envs/quadruped_terrain.py`)

The previous implementation measured torso uprightness using `abs(quat[0])` — the
absolute value of the quaternion w-component.  This quantity does not measure orientation
relative to gravity and produces incorrect uprightness decisions.

**Fix (PR #12):** replaced with `float(physics.torso_upright())`, the standard dm_control
helper that returns `xmat['torso', 'zz']` — the dot-product of the torso's local z-axis
with the world z-axis (1.0 = perfectly upright, −1.0 = fully inverted).

```python
# before (incorrect)
quat = physics.named.data.xquat["torso"]
torso_upright = float(abs(quat[0]))

# after (correct)
torso_upright = float(physics.torso_upright())
```

All 49 existing unit and integration tests continue to pass.

---

### Corrected Experimental Results

Because the oracle predicate was wrong in v1.0.0, the quantitative outcomes reported in
PAPER.md §8.2 and Table 5 were also wrong.  The corrected values, computed over seeds 0–9
with `physics.torso_upright()`, are:

| Condition | H_T | 95% CI | Violations ↓ | 95% CI | EXECUTE (%) |
|---|---|---|---|---|---|
| Baseline LLM | 0.65 | [0.47, 0.83] | 11.4 | [9.7, 13.1] | 100 |
| **LLM+Governor** | **0.65** | **[0.47, 0.83]** | **4.7** | **[3.0, 6.5]** | **30** |

Key findings (corrected):

- H_T (per §2.2 formula, all steps) is **identical** between conditions (0.65) because the
  governor does not change how often the LLM hallucinates.
- The governor's primary measurable benefit is a **59% reduction in unsafe-action
  violations** (11.4 → 4.7 per episode), driven by deterministic mode switching
  (30% EXECUTE / 64% VERIFY / 6% SAFE vs. 100% EXECUTE for baseline).

> **Correction note:** An earlier draft (v1.0.0) reported H_T = 0.41 → 0.24 (−41%) as
> the key embodied metric.  That figure was produced with the incorrect `abs(quat[0])`
> predicate and a non-standard H_T definition that excluded non-EXECUTE steps.

#### `VALIDATION_ANALYSIS.md` — replaced fabricated data with actual values (PR #13)

All benchmark figures in `VALIDATION_ANALYSIS.md` that were fabricated in v1.0.0 have been
replaced with actual measured values, including:

- Mode distribution (governor: 21.7% / 72.2% / 6.2%; ablation_a: 31.8% / 62.3% / 5.8%)
- Signal statistics (φ, ΔΦ, ΔI, ΔC, χ)
- Per-episode H_T, total rewards, and mode percentages for all four conditions
- Condition Comparison table and key observations
- Extended Condition Comparison: removed fabricated "Action Var" column and "Ablation B"
  row; corrected all remaining values
- Statistical Comparison: replaced fabricated H_T tests with actual violation
  Mann-Whitney U tests
- Governor overhead: replaced fabricated ms/CPU/Memory/Energy figures with the
  measured value of ~0.039 ms/step

#### Table 5 — added success rate column (PR #13)

A success rate column has been added to Table 5 in PAPER.md, with values computed from
the corrected oracle across seeds 0–9.

---

### Documentation Updates

#### PAPER.md

- **§8.2 oracle pipeline description** updated to reflect `physics.torso_upright()` and
  correct foot-contact geom names (`foot_front_left`, etc.).
- **Correction note** added below Table 5 explaining the v1.0.0 error.
- **Acknowledgements** section refined and expanded (multiple iterations) to include
  contributions from Google and to clarify assistance details.
- **Author information** and ORCID links updated for both authors.
- **References** section updated with additional citations.
- **Repository link and reviewer attribution** updated.
- **Citation formatting** corrected.

#### CITATION.cff

- Added new references corresponding to updated bibliography in PAPER.md.
- Updated `repository-code` URL.

#### README.md

- Acknowledgements updated to include final code review by Google.

---

### What Has Not Changed

- STG parameters: wR=0.30, wI=0.40, wC=0.30; α=0.25, β=0.35, γ=0.25, δ=0.15;
  τ₁=0.25, τ₂=0.55 (identical to synthetic testbed v2.2)
- All STG gating logic in `governor/spiral_time_governor.py`
- Experiment runner, mock LLM agent, analysis scripts
- Test suite (49 tests, all passing)
- Compatibility: Python 3.10+, MuJoCo 3.x, dm_control 1.0.14+

---

### Upgrade Notes

No API or configuration changes.  Re-run experiments with `run_experiment.py` and
`analysis/compute_metrics.py` to reproduce the corrected Table 5 values.

---

## v1.0.0 — Initial Embodied PoC Release

**Released:** 2026-02-28

### Overview

First public release of the Spiral-Time Governor MuJoCo proof-of-concept.
Demonstrates that the STG gating logic behaves consistently under real physics
dynamics, complementing the synthetic testbed results reported in the paper.

### What's Included

- **Spiral-Time Governor** — full deterministic implementation matching
  synthetic testbed v2.2 (wR=0.30, wI=0.40, wC=0.30; α=0.25, β=0.35,
  γ=0.25, δ=0.15; τ₁=0.25, τ₂=0.55)
- **MuJoCo environment wrapper** — dm_control quadruped "escape" task with
  ground-truth oracle derived from simulator state
- **Deterministic mock LLM agent** — no API keys required; controllable
  hallucination rate for reproducible experiments
- **Experiment runner** — supports baseline, governor, and ablation conditions
  across configurable seeds
- **Statistical analysis** — bootstrap CIs, Mann-Whitney U, Holm correction,
  and Cliff's δ; protocol matches manuscript §8.1 exactly
- **Test suite** — governor unit tests and environment integration tests

### Conditions Supported

| Condition | Ablation | Hallucination Prob |
|---|---|---|
| Baseline LLM | always_execute | 0.45 |
| LLM + Governor | none | 0.45 |
| Ablation A (δ=0) | no_delta | 0.45 |

### Known Limitations

- Uses dm_control quadruped "escape" as a proxy for ANYmal-class terrain tasks;
  custom task XML to follow in v1.1.0
- Oracle derived from simulator state; hardware transfer requires certified
  redundant monitors (manuscript §9)
- Mock LLM agent simulates hallucination probabilistically; real LLM integration
  planned for v2.0.0
- *(Fixed in v1.1.0)* `torso_upright` oracle used incorrect quaternion w-component
  predicate; quantitative results in §8.2 / Table 5 were affected
