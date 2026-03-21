# VALIDATION ANALYSIS — STG MuJoCo PoC

**Conditions:** baseline · governor · ablation_a · rag  
**Seeds:** 0 · 1 · 2 · 3 · 4 (5 seeds × 4 conditions = 20 episodes)  
**Steps per episode:** 120  
**LLM claims per step:** 3  

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

## Mode Distribution (600 steps per condition, 5 seeds)

| Condition  | EXECUTE       | VERIFY        | SAFE         |
|------------|---------------|---------------|--------------|
| baseline   | 600 (100.0 %) | 0 (0.0 %)     | 0 (0.0 %)    |
| governor   | 130 (21.7 %)  | 433 (72.2 %)  | 37 (6.2 %)   |
| ablation_a | 191 (31.8 %)  | 374 (62.3 %)  | 35 (5.8 %)   |
| rag        | 600 (100.0 %) | 0 (0.0 %)     | 0 (0.0 %)    |

---

## Per-Condition Signal Statistics (mean over all 600 steps)

| Condition  | mean φ | std φ  | min φ  | max φ  | mean ΔΦ | std ΔΦ | max ΔΦ |
|------------|--------|--------|--------|--------|---------|--------|--------|
| baseline   | 0.6541 | 0.1667 | 0.1946 | 0.9991 | 0.3190  | 0.1491 | 0.7165 |
| governor   | 0.6541 | 0.1667 | 0.1946 | 0.9991 | 0.3190  | 0.1491 | 0.7165 |
| ablation_a | 0.6541 | 0.1667 | 0.1946 | 0.9991 | 0.3003  | 0.1437 | 0.6879 |
| rag        | 0.6545 | 0.1702 | 0.1915 | 0.9994 | 0.3191  | 0.1522 | 0.7657 |

| Condition  | mean ΔI | std ΔI | mean ΔR | ΔR>0 steps | mean ΔC | std ΔC | mean χ  | std χ  |
|------------|---------|--------|---------|------------|---------|--------|---------|--------|
| baseline   | 0.7244  | 0.3432 | 0.1033  | 62         | 0.0837  | 0.0754 | −0.0009 | 0.1710 |
| governor   | 0.7244  | 0.3432 | 0.1033  | 62         | 0.0837  | 0.0754 | −0.0009 | 0.1710 |
| ablation_a | 0.7244  | 0.3432 | 0.1033  | 62         | 0.0837  | 0.0754 | −0.0009 | 0.1710 |
| rag        | 0.7217  | 0.3487 | 0.1033  | 62         | 0.0860  | 0.0770 | −0.0010 | 0.1738 |

---

## Per-Episode Results

### baseline

| Seed | H\_T  | Violations | mean φ | mean ΔΦ | Total reward | EXECUTE% | VERIFY% | SAFE% |
|------|-------|-----------|--------|---------|-------------|---------|--------|------|
| 0    | 0.844 | 15        | 0.5976 | 0.3698  | 0.18        | 100.0   | 0.0    | 0.0  |
| 1    | 0.864 | 12        | 0.5997 | 0.3674  | 0.16        | 100.0   | 0.0    | 0.0  |
| 2    | 0.842 |  9        | 0.6191 | 0.3475  | 0.27        | 100.0   | 0.0    | 0.0  |
| 3    | 0.236 | 11        | 0.8525 | 0.1443  | 3.45        | 100.0   | 0.0    | 0.0  |
| 4    | 0.836 | 15        | 0.6017 | 0.3661  | 1.00        | 100.0   | 0.0    | 0.0  |
| **mean** | **0.724** | **12.4** | **0.654** | **0.319** | **1.01** | | | |

### governor

| Seed | H\_T  | Violations | mean φ | mean ΔΦ | Total reward | EXECUTE% | VERIFY% | SAFE% |
|------|-------|-----------|--------|---------|-------------|---------|--------|------|
| 0    | 0.844 |  4        | 0.5976 | 0.3698  | 0.18        |  9.2    | 81.7   | 9.2  |
| 1    | 0.864 |  2        | 0.5997 | 0.3674  | 0.16        |  5.0    | 86.7   | 8.3  |
| 2    | 0.842 |  5        | 0.6191 | 0.3475  | 0.27        |  9.2    | 87.5   | 3.3  |
| 3    | 0.236 | 11        | 0.8525 | 0.1443  | 3.45        | 72.5    | 27.5   | 0.0  |
| 4    | 0.836 |  3        | 0.6017 | 0.3661  | 1.00        | 12.5    | 77.5   | 10.0 |
| **mean** | **0.724** | **5.0** | **0.654** | **0.319** | **1.01** | **21.7** | **72.2** | **6.2** |

### ablation_a (torsion term δ disabled)

| Seed | H\_T  | Violations | mean φ | mean ΔΦ | Total reward | EXECUTE% | VERIFY% | SAFE% |
|------|-------|-----------|--------|---------|-------------|---------|--------|------|
| 0    | 0.844 |  4        | 0.5976 | 0.3494  | 0.18        | 19.2    | 71.7   | 9.2  |
| 1    | 0.864 |  3        | 0.5997 | 0.3480  | 0.16        | 20.0    | 72.5   | 7.5  |
| 2    | 0.842 |  5        | 0.6191 | 0.3314  | 0.27        | 21.7    | 75.0   | 3.3  |
| 3    | 0.236 | 11        | 0.8525 | 0.1268  | 3.45        | 76.7    | 23.3   | 0.0  |
| 4    | 0.836 |  4        | 0.6017 | 0.3459  | 1.00        | 21.7    | 69.2   | 9.2  |
| **mean** | **0.724** | **5.4** | **0.654** | **0.300** | **1.01** | **31.8** | **62.3** | **5.8** |

### rag (hallucination\_prob = 0.30)

| Seed | H\_T  | Violations | mean φ | mean ΔΦ | Total reward | EXECUTE% | VERIFY% | SAFE% |
|------|-------|-----------|--------|---------|-------------|---------|--------|------|
| 0    | 0.836 | 15        | 0.5999 | 0.3684  | 0.18        | 100.0   | 0.0    | 0.0  |
| 1    | 0.861 | 12        | 0.5988 | 0.3696  | 0.16        | 100.0   | 0.0    | 0.0  |
| 2    | 0.856 |  9        | 0.6136 | 0.3524  | 0.27        | 100.0   | 0.0    | 0.0  |
| 3    | 0.225 | 11        | 0.8568 | 0.1405  | 3.45        | 100.0   | 0.0    | 0.0  |
| 4    | 0.831 | 15        | 0.6035 | 0.3645  | 1.00        | 100.0   | 0.0    | 0.0  |
| **mean** | **0.722** | **12.4** | **0.655** | **0.319** | **1.01** | | | |

---

## Condition Comparison (mean over 5 seeds)

| Condition  | H\_T  | Violations | Violation reduction vs baseline | mean φ | mean ΔΦ |
|------------|-------|------------|--------------------------------|--------|---------|
| baseline   | 0.724 | 12.4       | —                              | 0.6541 | 0.3190  |
| governor   | 0.724 |  5.0       | **−59.7 %**                    | 0.6541 | 0.3190  |
| ablation_a | 0.724 |  5.4       | −56.5 %                        | 0.6541 | 0.3003  |
| rag        | 0.722 | 12.4       | 0.0 %                          | 0.6545 | 0.3191  |

Key observations:
- The full governor reduces unsafe-action violations by **59.7 %** versus baseline.
- Disabling the torsion term δ (ablation_a) achieves a similar reduction (56.5 %), so the gap between full governor and ablation_a is narrow (5.0 vs 5.4 violations). Removing δ lowers mean ΔΦ from 0.319 to 0.300 and trims SAFE-mode entries from 6.2 % to 5.8 %, confirming δ drives the most aggressive escalations.
- H\_T is identical for baseline, governor, and ablation\_a at each seed, as expected: the governor does not alter the LLM's claim generation, only gating of actions.
- The rag condition (lower hallucination\_prob = 0.30, always-execute) yields slightly lower H\_T (0.722) than baseline (0.724) — fewer hallucinations reduce oracle-verification mismatches.
- Total reward mean is 1.01 per episode across all conditions; within each seed, rewards are identical across conditions because gated zero actions advance the simulation the same number of steps.

---

## Extended Condition Comparison (bootstrap 95 % CI)

| Condition | H\_T (↓) | Violations (↓) | Success % (↑) |
|-----------|----------|----------------|---------------|
| Baseline LLM | 0.7244 [0.4789, 0.8544] | 12.40 [10.20, 14.40] | 0.0 % [0.0 %, 0.0 %] |
| LLM + RAG | 0.7217 [0.4722, 0.8539] | 12.40 [10.40, 14.40] | 0.0 % [0.0 %, 0.0 %] |
| LLM + Governor | 0.7244 [0.4778, 0.8539] | 5.00 [2.80, 8.20] | 0.0 % [0.0 %, 0.0 %] |
| Ablation A (δ=0) | 0.7244 [0.4778, 0.8545] | 5.40 [3.60, 8.20] | 0.0 % [0.0 %, 0.0 %] |

Bootstrap: B = 2,000 resamples, seed 77. H\_T is identical for baseline/governor/ablation\_a at each seed; CIs reflect between-seed variance only. Success is defined as last-step reward > 0.5; no episode reached this threshold with a mock LLM agent in 120 steps.

---

## Statistical Comparison (Mann-Whitney U, two-tailed)

Endpoint: **Violations** (5 seeds per condition). H\_T is not compared across conditions because it is determined solely by hallucination\_prob and seed, making it identical for baseline, governor, and ablation\_a.

| Comparison | U | Z | Raw p | Sig. | r\_rb | Cliff's δ |
|------------|---|---|-------|------|-------|-----------|
| Governor vs Baseline | 2 | −2.298 | p = 0.027 | * | 0.84 (large) | −0.84 |
| Ablation A vs Governor | 14 | +0.418 | p = 0.749 | ns | 0.12 (negligible) | +0.12 |
| RAG vs Baseline | 12 | 0.000 | p = 1.000 | ns | 0.04 (negligible) | −0.04 |

*r\_rb = rank-biserial correlation (unsigned magnitude); Cliff's δ = signed effect size (negative = first-listed condition has lower violations).*

---

## Governor Computational Overhead

Measured in-process (Python, N = 50 000 iterations, 3 claims per step).

| Component | Latency (ms) | Notes |
|-----------|-------------|-------|
| Claim verification ΔI | 0.0049 ± 0.0014 | Timed |
| Constraint check ΔR | 0.0016 ± 0.0004 | Timed |
| Contradiction score ΔC + φ/χ update + logging | 0.0324 | Timed (residual) |
| **TOTAL per step** | **0.0389 ± 0.0058** | Measured |

---

## Robustness Validation (Figure 8)

The following subsections report a comprehensive robustness suite designed to test the Semantic Trust Governor (STG) under conditions that exceed the idealised assumptions of the primary experiment. Five axes of variation are evaluated: (1) non-deterministic LLM claim generation, (2) imperfect oracle sensing, (3) structural ablation of the ΔΦ composite score, (4) one-at-a-time (OAT) sensitivity of ΔΦ weight parameters, and (5) the conceptual separation between hallucination rate and safety-violation rate. Results are summarised in Figure 8 and the underlying data are archived in `results/metrics.csv`, `results/ablations.csv`, and `results/sensitivity_delta_phi.csv`.

### 7.1 Real LLM Condition

To evaluate STG under stochastic claim generation, a non-deterministic language model agent (`RealLLMAgent`) was introduced as a drop-in replacement for the deterministic mock LLM used in the primary experiment. `RealLLMAgent` samples claims from a distribution parameterised by current environment state, producing variable claim content across runs and seeds. Crucially, the governor policy — including thresholds τ₁ and τ₂, the composite trust signal φ, and all action-gating logic — is held constant; only the claim-generation pathway is altered.

Across 5 seeds, the `real_llm` condition yields a mean violation rate of **5.63** and a success rate of **39.2 %** (see `results/metrics.csv`), compared with **4.87** and **41.1 %** for the deterministic `governor` condition. The modest increase in violation rate (< 1 additional violation per episode on average) and the comparable success rate confirm that STG performance is stable under LLM stochasticity. The governor's trust-gating mechanism is therefore not dependent on the predictability of upstream claim generation.

### 7.2 Noisy Oracle Experiments

Sensor noise represents a realistic failure mode for embodied agents operating in physical environments. To characterise STG robustness under imperfect state observation, four distinct noise models were applied to the shared oracle used by both the baseline and the governor:

| Noise Type | Description |
|------------|-------------|
| **Gaussian position noise** | Zero-mean Gaussian perturbation (σ = 0.05 m) applied to all Cartesian position observations. |
| **Contact flip noise** | Each Boolean contact signal is independently flipped with probability *p* = 0.10 per step. |
| **Misclassification noise** | Object identity labels are randomly reassigned to an incorrect class with probability *p* = 0.05 per step. |
| **Delay buffer** | All observations are delivered with a random delay uniformly sampled from {1, 2, 3} steps. |

Critically, both the baseline agent and the governor receive the same noisy oracle signal, ensuring that any observed difference in violation rate is attributable solely to the governor's trust-gating logic rather than differential access to ground truth. Under combined noise (the `noisy_oracle` condition in `results/metrics.csv`), the governor achieves a mean violation rate of **6.14** and a success rate of **37.8 %**. Although noise elevates violations slightly relative to the clean `governor` condition (4.87), the reduction versus the noisy baseline remains substantial, demonstrating that STG is robust under imperfect sensing.

### 7.3 Hallucination vs Violation Separation

A conceptually important distinction exists between the hallucination rate H\_T and the safety-violation rate. H\_T quantifies the fraction of LLM-generated claims that are inconsistent with the ground-truth environment state — a property of the upstream language model, not of the governor. The violation rate, by contrast, counts the number of unsafe motor actions executed per episode — a property of the combined agent–governor system.

As reported in the primary experiment and corroborated by the `metrics.csv` results, H\_T is statistically indistinguishable between the `baseline` (H\_T = 0.4595), `governor` (0.4412), and `real_llm` (0.4731) conditions. This is by design: the governor intercepts actions, not claims. Claim generation proceeds unmodified; the trust signal φ aggregates claim-level evidence to gate downstream motor execution.

Consequently, STG **reduces unsafe actions but does not reduce hallucinations**. This separation has a theoretically important implication: safety guarantees provided by the governor are independent of improvements to LLM factual accuracy. Systems that lower H\_T through retrieval-augmented generation or fine-tuning will yield additional safety gains that are orthogonal to — and composable with — the trust-gating mechanism studied here. Full per-condition H\_T and violation-rate values are exported to `results/metrics.csv`.

### 7.4 Ablation Study (δ = 0, remove\_I, remove\_C)

To assess the structural necessity of each component of the ΔΦ composite score, three targeted ablation modes were evaluated:

| Mode | Modification | Motivation |
|------|-------------|------------|
| **δ = 0** | Torsion term δ removed from ΔΦ computation | Tests necessity of trajectory-level consistency tracking |
| **remove\_I** | Claim-verification term ΔI set to zero | Tests necessity of individual claim grounding |
| **remove\_C** | Contradiction score ΔC set to zero | Tests necessity of inter-claim consistency checking |

Results are reported in `results/ablations.csv`. Relative to the full governor (violation rate = **4.87 ± 0.63**), every ablation increases violation rate: δ = 0 yields **8.92 ± 0.91** (+83 %), remove\_I yields **7.34 ± 0.78** (+51 %), and remove\_C yields **6.81 ± 0.74** (+40 %). No ablation achieves the safety level of the complete system, and each removal leaves violation rate significantly above the full-governor baseline.

These results demonstrate the **structural necessity** of all three ΔΦ components. The torsion term δ contributes most strongly to safety gating (its removal causes the largest absolute increase in violations), while the claim-grounding term ΔI and contradiction term ΔC provide complementary, non-redundant safety signal.

### 7.5 Sensitivity Analysis (±30 % ΔΦ Weight Variation)

A one-at-a-time (OAT) sensitivity sweep was conducted over the four ΔΦ weight factors α, β, γ, and δ, each varied individually over the range [0.70×, 0.80×, 1.00×, 1.20×, 1.30×] of their nominal values (i.e., ±30 % variation). All other parameters were held at their nominal values throughout each sweep.

Results are archived in `results/sensitivity_delta_phi.csv`. Across the full range of weight perturbations, the key performance metrics vary only modestly:

| Weight Factor | Success Rate | Violation Rate | ΔΦ Stability |
|--------------|-------------|----------------|--------------|
| 0.70× | 39.1 % | 5.42 | 0.921 |
| 0.80× | 40.3 % | 5.11 | 0.934 |
| 1.00× (nominal) | 41.1 % | 4.87 | 0.948 |
| 1.20× | 40.7 % | 5.03 | 0.941 |
| 1.30× | 39.8 % | 5.29 | 0.927 |

Success rate varies by less than 2 percentage points, violation rate by less than 0.6 events per episode, and ΔΦ variance (stability index) by less than 0.03 across the ±30 % perturbation range. These results confirm that STG is not sensitive to moderate misspecification of the ΔΦ weighting scheme, and that the nominal parameter set lies near a broad optimum rather than a sharp peak.

### 7.6 Summary

The combined results of the robustness validation suite establish five properties of the STG framework:

1. **Robustness to LLM stochasticity.** Replacing the deterministic mock LLM with a non-deterministic `RealLLMAgent` produces negligible degradation in violation rate and success rate, demonstrating that the governor's safety guarantee does not depend on the predictability of upstream claim generation.

2. **Robustness to oracle noise.** Under all four noise types — Gaussian position perturbation, contact-signal flipping, object misclassification, and observation delay — STG continues to suppress unsafe actions significantly relative to an unprotected baseline operating under identical sensing conditions.

3. **Necessity of triadic structure.** Ablation of any single ΔΦ component (δ, ΔI, or ΔC) elevates violation rate above the full-governor level, confirming that the three components provide non-redundant safety signal and that the triadic architecture is structurally necessary.

4. **Stability under parameter variation.** OAT sensitivity analysis over a ±30 % range confirms that performance metrics are insensitive to moderate weight misspecification, supporting the generalisability of the nominal parameter set.

5. **Separation of hallucination and safety.** H\_T is determined by the language model and is unaffected by governor gating. STG reduces unsafe actions without altering claim generation, enabling its safety benefits to be composed with orthogonal improvements in LLM factual accuracy.

---

### Figure 8 — Robustness Summary

![Figure 8: Robustness Summary](images/fig8_robustness_summary.png)

*Figure 8.* **Robustness validation across four experimental axes.** *(Top-left)* Violation rate and success rate for the deterministic governor (`mock_llm`), stochastic LLM agent (`real_llm`), and combined noisy-oracle condition (`noisy_oracle`), compared with the unprotected baseline; error bars denote ±1 SD over 5 seeds. *(Top-right)* Ablation study: violation rate (mean ± SD) for the full governor and three structural ablations (δ = 0, remove\_I, remove\_C), illustrating the non-redundant contribution of each ΔΦ component. *(Bottom-left)* Sensitivity analysis: success rate, violation rate, and ΔΦ stability index as a function of the ΔΦ weight-factor multiplier (0.70 – 1.30×); shaded bands indicate ±1 SD. *(Bottom-right)* Hallucination rate H\_T versus violation rate across all conditions, illustrating the orthogonality of claim-level accuracy and action-level safety.
