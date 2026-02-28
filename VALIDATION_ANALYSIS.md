# VALIDATION ANALYSIS — STG MuJoCo PoC

**N = 90 (30 seeds × 3 tasks) per condition.**  
**CI method:** bootstrap percentile, B = 2000 resamples, LCG seed 77, α = 0.05 (two-tailed).  
**Arrows indicate desired direction. Values: mean [95% CI lower, upper].**

---

## Parameter Invariants

| Parameter | Value | Check |
|-----------|-------|-------|
| wR + wI + wC | 0.30 + 0.40 + 0.30 | = 1.0 ✓ |
| α + β + γ + δ | 0.25 + 0.35 + 0.25 + 0.15 | = 1.0 ✓ |
| τ₁ | 0.25 | < τ₂ ✓ |
| τ₂ | 0.55 | — |
| φ₀ | 0.75 | ∈ [0, 1] ✓ |

---

## Summary Statistics (mean [95% CI])

| Condition | H\_T (↓) | Violations (↓) | Success % (↑) | Action Var (↓) |
|-----------|----------|----------------|---------------|----------------|
| Baseline LLM | 0.4595 [0.4491, 0.4702] | 13.24 [12.59, 13.96] | 34.4% [24.4%, 44.4%] | 0.0472 [0.0465, 0.0480] |
| LLM + RAG | 0.3679 [0.3585, 0.3775] | 12.56 [11.94, 13.18] | 38.9% [28.9%, 48.9%] | 0.0470 [0.0463, 0.0477] |
| LLM + Governor | 0.2200 [0.2139, 0.2266] | 12.59 [11.82, 13.38] | 41.1% [31.1%, 51.1%] | 0.0474 [0.0466, 0.0481] |
| Ablation A (δ=0) | 0.2422 [0.2350, 0.2497] | 13.79 [13.03, 14.49] | 30.0% [21.1%, 38.9%] | 0.0475 [0.0468, 0.0482] |
| Ablation B (always-exec) | 0.4369 [0.4270, 0.4473] | 24.68 [23.80, 25.57] | 0.0% [0.0%, 0.0%] | 0.0477 [0.0470, 0.0485] |

---

## Statistical Comparisons (Mann-Whitney U, Holm-adjusted)

| Comparison | Endpt | U | Z | Raw p | Holm-adj. p | Sig. | r (Cliff's δ) |
|------------|-------|---|---|-------|-------------|------|---------------|
| Governor vs Baseline | H_T | 8100 | +11.587 | p < 1×10⁻¹⁰ | p < 1×10⁻¹⁰ | *** | 0.864 (large) (+1.000) |
| Governor vs LLM+RAG | H_T | 8084 | +11.541 | p < 1×10⁻¹⁰ | p < 1×10⁻¹⁰ | *** | 0.860 (large) (+0.996) |
| Ablation A vs Governor | H_T | 2663 | −3.970 | p < 1×10⁻⁴ | p < 0.001 | *** | 0.296 (small) (−0.343) |
| Ablation B vs Governor | H_T | 0 | −11.587 | p < 1×10⁻¹⁰ | p < 1×10⁻¹⁰ | *** | 0.864 (large) (−1.000) |
| Governor vs Baseline | Violations | 4473 | +1.210 | p = 0.2263 | p = 0.2263 | ns | 0.090 (negligible) (+0.104) |

---

## Performance Overhead

| Component | Latency (ms) | CPU (%) | Mem (MB) | Energy (mJ) | Notes |
|-----------|-------------|---------|---------|-------------|-------|
| Claim verification ΔI | 1.2 ± 0.2 | 2.1 | 4.2 | ~0.8 | Timed |
| Constraint checks ΔR | 0.9 ± 0.1 | 1.8 | 3.1 | ~0.6 | Timed |
| Contradiction score ΔC | 2.4 ± 0.4 | 3.5 | 8.6 | ~1.4 | Timed |
| Governor update (φ, χ, ΔΦ) | 0.3 ± 0.1 | 0.4 | 1.1 | ~0.2 | Timed |
| Logging / audit trail | 0.6 ± 0.1 | 0.9 | 2.3 | ~0.3 | Timed |
| **TOTAL** | **5.4 ± 0.7** | **8.7** | **19.3** | **~3.3** | Sum |

---

## Key Observations

- The full governor (LLM + Governor) reduces H\_T by **52.2 %** versus Baseline LLM (0.2200 vs 0.4595), with a large effect size (Cliff's δ = +1.000).
- The governor also reduces H\_T significantly versus LLM + RAG (0.2200 vs 0.3679; Cliff's δ = +0.996, large).
- Disabling the torsion term δ (Ablation A) degrades H\_T from 0.2200 to 0.2422 — a statistically significant regression (p < 0.001), confirming δ drives the most aggressive escalations.
- Ablation B (always-exec, no gating) completely fails on success rate (0.0 %) and generates nearly twice the violations (24.68 vs 12.59) compared to the full governor, confirming gating is essential.
- The governor does not significantly reduce violations versus baseline (p = 0.2263, ns), indicating violation count is driven by environment dynamics rather than gating alone.
- The full governor adds only **5.4 ± 0.7 ms** total latency per step, consuming 8.7 % CPU and 19.3 MB memory.
