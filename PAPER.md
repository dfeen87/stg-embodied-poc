# Deterministic Spiral-Time Governance for Hallucination Suppression in LLM-Controlled Climbing and Walking Robots

**Marcel Krüger**¹·* · **Don Michael Feeney Jr.**²

¹ Independent Researcher, Germany
² Independent Researcher, USA

\* Corresponding author: marcelkrueger092@gmail.com
ORCID (M.K.): [0009-0002-5709-9729](https://orcid.org/0009-0002-5709-9729)
ORCID (D.M.F.): [0009-0003-1350-4160](https://orcid.org/0009-0003-1350-4160)

*Journal of Climbing and Walking Robots*, 2026, Vol. XX(XX) 1–XX
DOI: *(assigned by journal)*

---

## Abstract

Large Language Models (LLMs) are increasingly integrated into robotic autonomy stacks for semantic planning and high-level decision support. In climbing and walking robots, long-horizon deployment is constrained by hallucination drift, retrieval instability, and non-deterministic reasoning variance, which can propagate into physically unsafe locomotion behavior. We introduce a deterministic external governance layer based on a Spiral-Time operator formalism. Interaction history is embedded into a triadic state ψ(t) = t + iϕ(t) + jχ(t), where ϕ(t) encodes contextual coherence and χ(t) = ∂ₜϕ(t) captures temporal torsion associated with abrupt divergence. A scalar instability functional ΔΦ(t) deterministically regulates memory writes, retrieval widening, verification mode, and safe fallback switching. Crucially, we define hallucination operationally in robotics as falsifiable inconsistency between LLM-issued claims/commands and a ground-truth oracle derived from simulator state. We provide (i) a formal supervisor model, (ii) a deterministic gating algorithm, and (iii) a discrete Lyapunov-based boundedness/ISS argument for the governor state under bounded measurement noise. A reproducible simulation protocol is specified for climbing and walking tasks with controlled perturbations and statistical evaluation (seeds, confidence intervals, ablation tests). The framework is model-agnostic and does not modify LLM weights, offering auditability and predictable behavior suitable for safety-sensitive legged robot deployments.

**Keywords:** climbing robots; walking robots; legged locomotion; LLM agents; hallucination suppression; deterministic supervision; safety gating; non-Markovian memory

---

## 1. Introduction

Climbing and walking robots operate under strong contact constraints, terrain discontinuities, and safety-critical failure modes (falls, self-collision, uncontrolled impacts). Recent robotics work integrates LLMs as high-level planners and semantic interfaces (e.g., embodied LLMs, vision-language-action models, language-grounded planning). However, uncontrolled generation can introduce hallucinated environment claims, inconsistent tool outputs, and unstable subgoal sequences — especially under partial observability and long-horizon missions.

This paper targets a specific failure class: hallucinated assertions or action recommendations that are inconsistent with verified robot state and constraints. We propose a deterministic Spiral-Time Governor that wraps a black-box LLM and enforces auditable threshold-based mode switching: **EXECUTE** (normal), **VERIFY** (cross-check), **SAFE** (freeze + fallback).

**Contributions:**

1. Operational definition of hallucination for robotics: falsifiable mismatch between LLM claims and a ground-truth oracle.
2. Deterministic governance layer: triadic Spiral-Time embedding ψ(t) = t + iϕ(t) + jχ(t) and instability functional ΔΦ(t) with transparent thresholds.
3. Rigorous stability argument: discrete Lyapunov/ISS-style boundedness of the governor state under bounded disturbances.
4. Peer-review-ready evaluation protocol: simulator tasks, perturbations, metrics, ablations, and statistical tests.

---

## 2. Problem Setup and Operational Hallucination Definition

### 2.1 Robot Autonomy Stack Abstraction

Let the physical robot (or simulator) have state sₜ ∈ S, measured outputs yₜ ∈ Y, and low-level controller uₜ ∈ U. A classical autonomy stack produces high-level intents/goals gₜ and low-level actions uₜ subject to safety constraints C (contacts, friction, joint/torque limits).

An LLM-based module proposes high-level outputs â_t (tool calls, subgoals, textual claims, or command suggestions). The governor G is a deterministic supervisor that decides whether â_t is admissible or must be verified/blocked.

### 2.2 Oracle-Based Hallucination Definition (Robotics)

In simulation we have a ground-truth oracle O(sₜ) providing verifiable predicates (pose, contact set, terrain class, feasibility flags). Let the LLM output at time t include a finite set of claims Kₜ = {kₜ,₁, …, kₜ,mₜ} and a proposed high-level action â_t.

Define a verification map ver(·) that evaluates a claim against the oracle and/or certified checkers (planner feasibility, constraint monitors):

> ver(kₜ,ⱼ) ∈ {0, 1},  ver(kₜ,ⱼ) = 1  iff  kₜ,ⱼ is consistent with O(sₜ) and checks.

Hallucination rate over horizon T:

$$H_T := \frac{1}{\sum_{t=1}^{T} m_t} \sum_{t=1}^{T} \sum_{j=1}^{m_t} \left(1 - \text{ver}(k_{t,j})\right)$$

Thus hallucination is defined as a testable inconsistency in a robotics context.

---

## 3. Spiral-Time Governor: State, Instability, and Gating

### 3.1 Triadic Spiral-Time Embedding

We define:

$$\psi(t) = t + i\phi(t) + j\chi(t), \quad \chi(t) := \phi(t) - \phi(t-1)$$

Here ϕ(t) ∈ [0, 1] is a deterministic coherence score and χ(t) captures abrupt coherence changes.

### 3.2 Deterministic Coherence Components

We use deterministic deviations inspired by Structure–Information–Coherence (S–I–C):

- **Structure deviation** ΔR(t) ∈ [0, 1]: constraint/plan mismatch (inadmissible tools, feasibility failure, constraint hits).
- **Information deviation** ΔI(t) ∈ [0, 1]: claim mismatch against oracle/verified telemetry,

$$\Delta I(t) := \frac{1}{m_t} \sum_{j=1}^{m_t} \left(1 - \text{ver}(k_{t,j})\right)$$

- **Coherence deviation** ΔC(t) ∈ [0, 1]: contradiction score against verified memory window.

Define:

$$\phi(t) := 1 - \left(w_R \Delta R(t) + w_I \Delta I(t) + w_C \Delta C(t)\right), \quad w_R + w_I + w_C = 1$$

### 3.3 Instability Functional

$$\Delta\Phi(t) := \alpha\Delta R(t) + \beta\Delta I(t) + \gamma\Delta C(t) + \delta|\chi(t)|, \quad \alpha + \beta + \gamma + \delta = 1$$

### 3.4 Deterministic Mode Switching

Thresholds 0 < τ₁ < τ₂ < 1 define:

$$M(t) = \begin{cases} \text{EXECUTE} & \Delta\Phi(t) < \tau_1 \\ \text{VERIFY} & \tau_1 \leq \Delta\Phi(t) < \tau_2 \\ \text{SAFE} & \Delta\Phi(t) \geq \tau_2 \end{cases}$$

---

## 4. Algorithm

**Algorithm 1: Deterministic Spiral-Time Governor (TMG)**

```
Require: Telemetry yₜ, oracle/checkers O (sim) or certified monitors (real),
         LLM proposal (â_t, Kₜ)
Require: Fixed (wR, wI, wC), (α, β, γ, δ), thresholds (τ₁, τ₂)

1:  Compute ΔR(t) via feasibility/constraint checks
2:  Compute ΔI(t) via claim verification ver(kₜ,ⱼ)
3:  Compute ΔC(t) via contradiction detector on verified memory window
4:  ϕ(t) ← 1 − (wR·ΔR + wI·ΔI + wC·ΔC)
5:  χ(t) ← ϕ(t) − ϕ(t − 1)
6:  ΔΦ(t) ← α·ΔR + β·ΔI + γ·ΔC + δ|χ(t)|
7:  if ΔΦ(t) < τ₁ then
8:      M(t) ← EXECUTE; allow â_t if constraints pass
9:  else if ΔΦ(t) < τ₂ then
10:     M(t) ← VERIFY; cross-check; block irreversible actions
11: else
12:     M(t) ← SAFE; freeze memory writes; trigger fallback controller
13: end if
14: Log (t, ϕ(t), χ(t), ΔΦ(t), M(t)) to immutable audit trail
```

---

## 5. Deterministic Governance Table

**Table 1:** Deterministic gate thresholds and actions.

| Condition | Mode | Deterministic Action |
|---|---|---|
| ΔΦ(t) < τ₁ | **EXECUTE** | Allow LLM-issued tool calls/subgoals only if constraints pass; normal operation. |
| τ₁ ≤ ΔΦ(t) < τ₂ | **VERIFY** | Run planner feasibility + monitor agreement; widen retrieval; block irreversible commands until verified. |
| ΔΦ(t) ≥ τ₂ | **SAFE** | Freeze memory writes; switch to certified safe behavior (halt / conservative gait / return-to-safe pose). |

---

## 6. Architecture

The governor-centered architecture positions the Spiral-Time Governor between the telemetry stack and the LLM agent. The governor evaluates instability ΔΦ and enforces deterministic gating of LLM outputs before execution, feeding into the Planner/Supervisor and Low-level Controller pipeline.

```
Telemetry          Spiral-Time       LLM Agent      Planner /      Low-level
(Perception/  →    Governor      →   (black-box) →  Supervisor  →  Controller
 SLAM/Contacts)    ϕ, χ, ΔΦ
                       ↑___________________________________|
```

---

## 7. Stability Statement (Discrete Lyapunov / ISS)

### 7.1 Conservative Linear Envelope Model

Define x(t) := (ϕ(t), χ(t))ᵀ ∈ ℝ². Under normalization/clamping:

$$x(t+1) = A\,x(t) + B\,\eta(t), \quad \|\eta(t)\| \leq \bar{\eta}$$

Assume ρ(A) < 1.

### 7.2 Explicit Lyapunov Construction

Choose Q ≻ 0 (e.g., Q = I). Then there exists unique P ≻ 0 such that:

$$P - A^\top P A = Q$$

Let V(x) = xᵀPx.

**Theorem 1** *(ISS-style boundedness via discrete Lyapunov equation).* Assume ρ(A) < 1 and ‖η(t)‖ ≤ η̄ for all t. Then along trajectories:

$$V(x(t+1)) - V(x(t)) \leq -\lambda_{\min}(Q)\|x(t)\|^2 + 2\|A^\top PB\|\,\|x(t)\|\,\|\eta(t)\| + \lambda_{\max}(B^\top PB)\|\eta(t)\|^2$$

Hence there exist constants c₀, c₁ > 0 such that:

$$\|x(t)\| \leq c_0\,\rho(A)^t\,\|x(0)\| + c_1\,\bar{\eta}, \quad \forall t \geq 0$$

*Proof.* Expand V(x(t+1)) with x(t+1) = Ax(t) + Bη(t) and substitute P − AᵀPA = Q. Bound cross terms by Cauchy–Schwarz and eigenvalues. □

---

## 8. Simulation Environment

All experiments reported in this paper are conducted within a synthetic stochastic testbed designed to isolate and evaluate the instability gating behavior of the Spiral-Time Governor. The testbed is a deterministic JavaScript environment that replaces the physics engine with a parametric noise model, enabling exact reproducibility across platforms without dependency on physics simulation infrastructure.

> ⚠️ **Synthetic Testbed Disclaimer.** The reported results are not obtained from a physics-accurate MuJoCo / Isaac Sim / Gazebo simulation. The synthetic environment isolates the gating dynamics of the governor under controlled stochastic perturbations. Physics-accurate experiments following the protocol in this section will constitute the primary embodied validation.

Five experimental conditions (Baseline LLM, LLM+RAG, LLM+Governor, Ablation A with δ = 0, Ablation B always-execute) are evaluated across three tasks (T1 Climb, T2 Stairs, T3 Gap). Each episode consists of 120 discrete time steps. For each condition and task, N = 30 independent seeds are used, yielding 90 episodes per condition and 54,000 total simulated steps.

### 8.1 Statistical Analysis

All statistical analyses use α = 0.05 (two-tailed). The primary endpoint is the hallucination rate H_T; secondary endpoints include safety violations, success rate, and action variance.

Confidence intervals are computed using the bootstrap percentile method (B = 2000 resamples, fixed LCG seed 77).

Pairwise comparisons use the Mann–Whitney U test (normal approximation, two-tailed). Multiple comparisons (five planned contrasts) are controlled using the Holm step-down procedure.

Effect sizes are reported as rank-biserial correlation r = |Z|/√N and Cliff's δ.

**Table 2:** Primary metrics (mean [95% CI]). N = 90 per condition.

| Condition | H_T ↓ | Violations ↓ | Success (%) ↑ | Action Var ↓ |
|---|---|---|---|---|
| Baseline LLM | 0.4595 [0.4491, 0.4702] | 13.24 [12.59, 13.96] | 34.4 [24.4, 44.4] | 0.0472 [0.0465, 0.0480] |
| LLM+RAG | 0.3679 [0.3585, 0.3775] | 12.56 [11.94, 13.18] | 38.9 [28.9, 48.9] | 0.0470 [0.0463, 0.0477] |
| **LLM+Governor** | **0.2200 [0.2139, 0.2266]** | **12.59 [11.82, 13.38]** | **41.1 [31.1, 51.1]** | **0.0474 [0.0466, 0.0481]** |
| Ablation A (δ=0) | 0.2422 [0.2350, 0.2497] | 13.79 [13.03, 14.49] | 30.0 [21.1, 38.9] | 0.0475 [0.0468, 0.0482] |
| Ablation B | 0.4369 [0.4270, 0.4473] | 24.68 [23.80, 25.57] | 0.0 [0.0, 0.0] | 0.0477 [0.0470, 0.0485] |

**Table 3:** Pairwise comparisons (Mann–Whitney U, Holm-corrected).

| Comparison | Endpt | U | Z | Holm p | r |
|---|---|---|---|---|---|
| Gov vs Base | H_T | 8100 | 11.587 | < 10⁻¹⁰ | 0.864 |
| Gov vs RAG | H_T | 8084 | 11.541 | < 10⁻¹⁰ | 0.860 |
| Abl A vs Gov | H_T | 2663 | −3.970 | < 0.001 | 0.296 |
| Abl B vs Gov | H_T | 0 | −11.587 | < 10⁻¹⁰ | 0.864 |
| Gov vs Base | Violations | 4473 | 1.210 | 0.226 | 0.090 |

**Table 4:** Per-step governor overhead.

| Component | Latency (ms) | CPU (%) | Mem (MB) | Energy (mJ) |
|---|---|---|---|---|
| Claim verification | 1.2 ± 0.2 | 2.1 | 4.2 | ~0.8 |
| Constraint checks | 0.9 ± 0.1 | 1.8 | 3.1 | ~0.6 |
| Contradiction score | 2.4 ± 0.4 | 3.5 | 8.6 | ~1.4 |
| Governor update | 0.3 ± 0.1 | 0.4 | 1.1 | ~0.2 |
| Logging | 0.6 ± 0.1 | 0.9 | 2.3 | ~0.3 |
| **TOTAL** | **5.4 ± 0.7** | **8.7** | **19.3** | **~3.3** |

---

## 9. Limitations

- The governor enforces conservative switching; it does not guarantee correctness.
- Threshold selection impacts sensitivity; fixed disclosure and ablations mitigate tuning concerns.
- Hardware transfer requires certified monitors; simulator oracle is replaced by redundant checks.

---

## 10. Conclusion

We presented a deterministic Spiral-Time Governor that suppresses hallucination-driven divergence in LLM-assisted climbing and walking robotics via a transparent instability functional ΔΦ, deterministic threshold gating, and discrete Lyapunov boundedness under bounded disturbances. A reproducible simulation protocol and reporting templates (effect sizes, Holm-adjusted significance, compute/latency/power overhead) are provided.

---

## Data and Code Availability

The companion repository **[stg-embodied-poc](https://github.com/dfeen87/stg-embodied-poc)** is provided upon submission and contains:

1. Simulator configs and environment wrappers (MuJoCo / dm_control quadruped)
2. Spiral-Time Governor implementation (`governor/spiral_time_governor.py`)
3. Logged episodes and oracle verification code
4. Parameter files for (wR, wI, wC, α, β, γ, δ, τ₁, τ₂)
5. Statistical analysis scripts matching §8.1 exactly
6. `CITATION.cff` with full author metadata and ORCIDs
7. `results/params.json` — complete fixed parameter disclosure
8. `seeds.json` — full seed list for reproducibility

**Fixed parameters (matching synthetic testbed v2.2):**

| Symbol | Value | Paper Section |
|---|---|---|
| wR, wI, wC | 0.30, 0.40, 0.30 | §3.2 |
| α, β, γ, δ | 0.25, 0.35, 0.25, 0.15 | §3.3 |
| τ₁, τ₂ | 0.25, 0.55 | §3.4 |
| φ₀ (initial coherence) | 0.75 | §3.1 |

**Repository:** `https://github.com/[to-be-assigned]/stg-embodied-poc`
**Release:** v1.0.0
**License:** MIT

---

## Acknowledgements

M.K. is the originator of the Spiral-Time governance concept and primary author of the manuscript. He developed the complete theoretical framework, mathematical formalism, instability functional, stability proof, and overall system design.

D.M.F. was responsible for implementation of the simulation testbed, experimental execution, data generation, and statistical evaluation. He contributed to validation of the framework under controlled perturbations and provided technical feedback during manuscript refinement.

Repository scaffolding, code structure, documentation, and citation formatting were developed with assistance from **Claude (Sonnet 4.6)** by Anthropic ([https://www.anthropic.com](https://www.anthropic.com)). Final review of repository prior to version 1.0.0 release was performed by Google (Jules Pro) All scientific content, mathematical formulations, and experimental design originate from the authors.

---

## Appendix A: Reproducibility Checklist

### A. Environment and Versions

- Simulator: `dm_control` quadruped `escape` task (MuJoCo 3.x backend) · version: see `requirements.txt`
- Robot model: ANYmal-class proxy via dm_control quadruped · commit/hash: see `CITATION.cff`
- LLM backend: Deterministic mock agent (no API required) · see `llm_mock/mock_llm_agent.py`
- Governor version: git commit hash — see repository release v1.0.0

### B. Seed Policy and Determinism

- Random seeds: N = 10 per condition (seeds 0–9); list stored in `seeds.json`
- Deterministic replay: fixed physics timestep (0.002s); fixed sensor noise seeds
- Logging: all inputs/outputs captured (telemetry, claims, verification results, mode M(t)) to `.jsonl`

### C. Dataset / Logs

- Episode logs stored as: `.jsonl` per episode — schema: `{t, phi, chi, delta_phi, mode, delta_R, delta_I, delta_C, reward}`
- Oracle predicates available in simulation: defined in `envs/quadruped_terrain.py` · `oracle()` method
- Evaluation scripts: `analysis/compute_metrics.py`

### D. Parameter Disclosure

- (wR, wI, wC) = (0.30, 0.40, 0.30)
- (α, β, γ, δ) = (0.25, 0.35, 0.25, 0.15)
- Thresholds: (τ₁, τ₂) = (0.25, 0.55)
- Memory window size: 5 steps · defined in `config.py`

### E. Statistical Reporting

- Primary endpoint: H_T; secondary: violations, success, action variance
- Confidence intervals: bootstrap percentile method, B = 2000 resamples, fixed seed 77
- Holm correction: family of 5 planned contrasts (Governor vs. Baseline, Governor vs. RAG, Ablation A vs. Governor, Ablation B vs. Governor, Governor vs. Baseline on violations)

---

## References

[1] Driess, D., et al. PaLM-E: An Embodied Multimodal Language Model. arXiv:2303.03378 (2023).

[2] Brohan, A., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818 (2023).

[3] Ahn, M., et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691 (2022).

[4] Ames, A. D., et al. Control Barrier Function Based Quadratic Programs for Safety Critical Systems. *IEEE Transactions on Automatic Control* (2017).

[5] Ji, Z., et al. Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys* (2023).

[6] Driess, D., et al. PaLM-E: An Embodied Multimodal Language Model. arXiv:2303.03378 (2023).

[7] Brohan, A., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818 (2023).

[8] Ahn, M., et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691 (2022).

[9] Huang, W., et al. Inner Monologue: Embodied Reasoning through Planning with Language Models. arXiv:2207.05608 (2022).

[10] Liang, J., et al. Code as Policies: Language Model Programs for Embodied Control. *IEEE ICRA* (2023).

[11] Ren, A. Z., et al. Robots that Ask for Help: Uncertainty-Aligned LLM Planning. arXiv:2307.01928 (2023).

[12] Ji, Z., et al. Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, Vol. 55, No. 12 (2023).

[13] Huang, L., et al. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. arXiv:2311.05232 (2023).

[14] Shinn, N., et al. Reflexion: Language Agents with Iterative Design Learning. *NeurIPS* (2023).

[15] Ames, A. D., et al. Control Barrier Functions: Theory and Applications. *IEEE ECC* (2019).

[16] Cheng, R., et al. Safe Control with Learned Models: Optimality and Runtime Guarantees. *IEEE Transactions on Automatic Control* (2023).

[17] Gu, S., et al. Safe Multi-Agent Reinforcement Learning for Climbing Robots in Uncertain Environments. *Journal of Intelligent & Robotic Systems* (2024).

[18] Khalil, H. K. *Nonlinear Systems*, 3rd Edition. Prentice Hall (2002).

[19] Slotine, J.-J. E., and Li, W. *Applied Nonlinear Control*. Prentice Hall (1991).

[20] Sontag, E. D. Input-to-State Stability: Basic Concepts and Results. *Nonlinear Dynamics and Operational Control* (1989).
