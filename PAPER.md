# Deterministic Spiral-Time Governance for Hallucination Suppression in LLM-Controlled Climbing and Walking Robots

Marcel Krüger¹

Don Michael Feeney Jr²

¹ Independent Researcher, Germany
² Independent Researcher, USA

Corresponding author: [marcelkrueger092@gmail.com](mailto:marcelkrueger092@gmail.com)
ORCID (M.K.): [0009-0002-5709-9729](https://orcid.org/0009-0002-5709-9729)
ORCID (D.M.F.): [0009-0003-1350-4160](https://orcid.org/0009-0003-1350-4160)

*Journal of Climbing and Walking Robots* · 2026 · Vol. XX(XX) · pp. 1–XX
DOI: *(assigned by journal)*

---

## Abstract

Large Language Models (LLMs) are increasingly integrated into robotic autonomy stacks for semantic planning and high-level decision support. In climbing and walking robots, long-horizon deployment is constrained by hallucination drift, retrieval instability, and non-deterministic reasoning variance, which can propagate into physically unsafe locomotion behavior.

We introduce a deterministic external governance layer based on a Spiral-Time operator formalism. Interaction history is embedded into a triadic state ψ(t) = t + iϕ(t) + jχ(t), where ϕ(t) encodes contextual coherence and χ(t) = ∂ₜϕ(t) captures temporal torsion associated with abrupt divergence. A scalar instability functional ΔΦ(t) deterministically regulates memory writes, retrieval widening, verification mode, and safe fallback switching.

Crucially, we define hallucination operationally in robotics as **falsifiable inconsistency** between LLM-issued claims/commands and a ground-truth oracle derived from simulator state. We provide (i) a formal supervisor model, (ii) a deterministic gating algorithm, and (iii) a discrete Lyapunov-based boundedness/ISS argument for the governor state under bounded measurement noise. A reproducible simulation protocol is specified for climbing and walking tasks with controlled perturbations and statistical evaluation (seeds, confidence intervals, ablation tests).

In addition to the synthetic stochastic evaluation, we provide a **minimal physics-grounded transfer validation** using a MuJoCo-based quadruped environment. All governor parameters are kept identical to the synthetic setting without retuning. The embodied validation reproduces a quantitative reduction in hallucination rate (H_T: 0.41 → 0.24, ≈41% relative reduction) and deterministic mode switching behavior under contact-rich dynamics.

The framework is model-agnostic and does not modify LLM weights, offering auditability and predictable behavior suitable for safety-sensitive legged robot deployments.

**Keywords:** legged robotics · LLM-based control · hallucination detection · deterministic safety supervision · MuJoCo validation

---

## 1 · Introduction

Climbing and walking robots operate under strong contact constraints, terrain discontinuities, and safety-critical failure modes (falls, self-collision, uncontrolled impacts). The stability perspective adopted here is conceptually aligned with barrier-function and Lyapunov-based safety formulations in control theory [1], although the present governor operates at the **semantic decision layer** rather than continuous torque control.

Recent robotics research increasingly integrates LLMs as high-level planners, semantic interfaces, and vision-language-action controllers. While such models enable flexible task abstraction and semantic reasoning, their generative nature introduces instability risks: hallucinated environment claims, inconsistent tool outputs, and non-deterministic subgoal sequences can propagate into physically unsafe locomotion behavior — especially under partial observability and long-horizon deployment.

Existing mitigation strategies typically rely on confidence scoring, retrieval augmentation, or post-hoc verification. However, these approaches remain **probabilistic** and do not provide deterministic guarantees on mode switching or bounded supervisory dynamics.

This paper targets a specific failure class: hallucinated assertions or action recommendations that are inconsistent with verified robot state and constraints. We propose a **deterministic Spiral-Time Governor** that wraps a black-box LLM and enforces auditable threshold-based mode switching:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   EXECUTE   │    │    VERIFY    │    │     SAFE     │
│             │    │              │    │              │
│ Normal      │    │ Cross-check  │    │ Freeze +     │
│ operation   │    │ & constraint │    │ certified    │
│             │    │ validation   │    │ fallback     │
└─────────────┘    └──────────────┘    └──────────────┘
  ΔΦ(t) < τ₁       τ₁ ≤ ΔΦ(t) < τ₂     ΔΦ(t) ≥ τ₂
```

The governor does not modify LLM weights and is model-agnostic. It provides a transparent instability functional that regulates execution, verification, and fallback behavior through deterministic thresholds.

### Contributions

1. **Operational hallucination definition** for robotics: falsifiable mismatch between LLM claims and a ground-truth oracle derived from simulator state.
2. **Deterministic governance layer**: triadic Spiral-Time embedding ψ(t) = t + iϕ(t) + jχ(t) and instability functional ΔΦ(t) with transparent thresholds and auditable switching logic.
3. **Rigorous stability argument**: discrete Lyapunov/ISS-style boundedness of the governor state under bounded disturbances.
4. **Two-layer evaluation protocol**: controlled synthetic stochastic testbed and complementary MuJoCo-based physics-grounded validation, including seeds, ablations, statistical testing, and compute overhead analysis.

---

## 2 · Problem Setup and Operational Hallucination Definition

### 2.1 · Robot Autonomy Stack Abstraction

Let the physical robot (or simulator) have state sₜ ∈ 𝒮, measured outputs yₜ ∈ 𝒴, and low-level controller uₜ ∈ 𝒰. A classical autonomy stack produces high-level intents/goals gₜ and low-level actions uₜ subject to safety constraints 𝒞 (contacts, friction, joint/torque limits).

An LLM-based module proposes high-level outputs â_t (tool calls, subgoals, textual claims, or command suggestions). The governor 𝒢 is a **deterministic supervisor** that decides whether â_t is admissible or must be verified/blocked.

```
  Physical Robot / Simulator
  ┌────────────────────────────────────────────────┐
  │  sₜ ∈ 𝒮  ──→  yₜ ∈ 𝒴  ──→  uₜ ∈ 𝒰          │
  │                                    ↑            │
  │         LLM Module                 │            │
  │    â_t (claims, subgoals)          │            │
  │           │                        │            │
  │           ↓                        │            │
  │       Governor 𝒢  ─── admissible? ─┘            │
  │    (deterministic supervisor)                   │
  └────────────────────────────────────────────────┘
```

### 2.2 · Oracle-Based Hallucination Definition

In simulation we have a ground-truth oracle 𝒪(sₜ) providing verifiable predicates (pose, contact set, terrain class, feasibility flags). Let the LLM output at time t include a finite set of claims Kₜ = {kₜ,₁, …, kₜ,mₜ} and a proposed action â_t.

Define a verification map ver(·) that evaluates a claim against the oracle and/or certified checkers:

> **ver(kₜ,ⱼ) ∈ {0, 1}** — equals 1 iff kₜ,ⱼ is consistent with 𝒪(sₜ) and certified checks.

**Hallucination rate over horizon T:**

$$H_T \;:=\; \frac{1}{\displaystyle\sum_{t=1}^{T} m_t} \;\sum_{t=1}^{T}\;\sum_{j=1}^{m_t} \bigl(1 - \mathrm{ver}(k_{t,j})\bigr)$$

Thus hallucination is defined as a **testable inconsistency** in a robotics context — not a probabilistic confidence estimate.

---

## 3 · Spiral-Time Governor: State, Instability, and Gating

### 3.1 · Triadic Spiral-Time Embedding

```
         Imaginary axes
              │  jχ(t)
              │   ↑
              │   │  · ψ(t)
              │   │ ╱
   ────────────┼──╱──────────────→  iϕ(t)
              │╱
              ╱
             ╱ t (real / time axis)
```

We define the triadic state:

$$\psi(t) \;=\; t \;+\; i\,\phi(t) \;+\; j\,\chi(t), \qquad \chi(t) \;:=\; \phi(t) - \phi(t-1)$$

Here **ϕ(t) ∈ [0, 1]** is a deterministic coherence score and **χ(t)** captures abrupt coherence changes (temporal torsion).

### 3.2 · Deterministic Coherence Components

We use deterministic deviations inspired by the Structure–Information–Coherence (S–I–C) framework:

| Component | Symbol | Range | Captures |
|---|---|---|---|
| Structure deviation | ΔR(t) | [0, 1] | Constraint/plan mismatch, inadmissible tools, feasibility failures |
| Information deviation | ΔI(t) | [0, 1] | Claim mismatch against oracle/telemetry |
| Coherence deviation | ΔC(t) | [0, 1] | Contradiction score vs. verified memory window |

**Information deviation** is defined directly from the verification map:

$$\Delta I(t) \;:=\; \frac{1}{m_t}\sum_{j=1}^{m_t}\bigl(1 - \mathrm{ver}(k_{t,j})\bigr)$$

**Coherence score:**

$$\phi(t) \;:=\; 1 \;-\;\bigl(w_R\,\Delta R(t) + w_I\,\Delta I(t) + w_C\,\Delta C(t)\bigr), \qquad w_R + w_I + w_C = 1$$

### 3.3 · Instability Functional

$$\boxed{\Delta\Phi(t) \;:=\; \alpha\,\Delta R(t) \;+\; \beta\,\Delta I(t) \;+\; \gamma\,\Delta C(t) \;+\; \delta\,|\chi(t)|, \qquad \alpha+\beta+\gamma+\delta = 1}$$

### 3.4 · Deterministic Mode Switching

$$M(t) \;=\; \begin{cases} \textbf{EXECUTE} & \Delta\Phi(t) < \tau_1 \\ \textbf{VERIFY} & \tau_1 \leq \Delta\Phi(t) < \tau_2 \\ \textbf{SAFE} & \Delta\Phi(t) \geq \tau_2 \end{cases}$$

**Fixed parameters (v2.2):**

| Symbol | Value | Role |
|---|---|---|
| wR, wI, wC | 0.30, 0.40, 0.30 | Coherence weights |
| α, β, γ, δ | 0.25, 0.35, 0.25, 0.15 | ΔΦ weights |
| τ₁, τ₂ | 0.25, 0.55 | Mode thresholds |
| ϕ₀ | 0.75 | Initial coherence |

---

## 4 · Algorithm

**Algorithm 1: Deterministic Spiral-Time Governor (STG)**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Require: Telemetry yₜ, oracle 𝒪 (sim) or certified monitors (real)
 Require: LLM proposal (â_t, Kₜ)
 Require: Fixed (wR, wI, wC), (α, β, γ, δ), thresholds (τ₁, τ₂)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1:  Compute ΔR(t)  ◀  feasibility/constraint checks
  2:  Compute ΔI(t)  ◀  claim verification ver(kₜ,ⱼ) vs oracle
  3:  Compute ΔC(t)  ◀  contradiction detector on memory window
  4:  ϕ(t)  ←  1 − (wR·ΔR + wI·ΔI + wC·ΔC)
  5:  χ(t)  ←  ϕ(t) − ϕ(t − 1)
  6:  ΔΦ(t) ←  α·ΔR + β·ΔI + γ·ΔC + δ·|χ(t)|
  7:  ┌─ if ΔΦ(t) < τ₁ ──────────────────────────────────────┐
  8:  │    M(t) ← EXECUTE ;  allow â_t if constraints pass    │
  9:  ├─ else if ΔΦ(t) < τ₂ ──────────────────────────────────┤
 10:  │    M(t) ← VERIFY  ;  cross-check; block irreversible   │
 11:  ├─ else ──────────────────────────────────────────────────┤
 12:  │    M(t) ← SAFE    ;  freeze writes; trigger fallback    │
 13:  └───────────────────────────────────────────────────────-─┘
 14:  Log (t, ϕ(t), χ(t), ΔΦ(t), M(t))  →  immutable audit trail
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 5 · Deterministic Governance Table

**Table 1:** Gate thresholds and deterministic actions.

| Condition | Mode | Deterministic Action |
|---|:---:|---|
| ΔΦ(t) < τ₁ | **EXECUTE** | Allow LLM-issued tool calls/subgoals only if constraints pass; normal operation. |
| τ₁ ≤ ΔΦ(t) < τ₂ | **VERIFY** | Run planner feasibility + monitor agreement; widen retrieval; block irreversible commands until verified. |
| ΔΦ(t) ≥ τ₂ | **SAFE** | Freeze memory writes; switch to certified safe behavior (halt / conservative gait / return-to-safe pose). |

---

## 6 · Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│   │  Telemetry   │    │  Spiral-Time     │    │   LLM Agent     │  │
│   │              │───▶│  Governor        │───▶│   (black-box)   │  │
│   │  Perception  │    │                  │    │                 │  │
│   │  SLAM        │    │  ϕ(t)  coherence │    └────────┬────────┘  │
│   │  Contacts    │    │  χ(t)  torsion   │             │           │
│   └──────────────┘    │  ΔΦ(t) instabil. │             ▼           │
│                       │                  │    ┌─────────────────┐  │
│                       │  ┌────────────┐  │    │    Planner /    │  │
│                       │  │  EXECUTE   │  │    │   Supervisor    │  │
│                       │  │  VERIFY    │  │    └────────┬────────┘  │
│                       │  │  SAFE      │  │             │           │
│                       │  └────────────┘  │             ▼           │
│                       └──────────────────┘    ┌─────────────────┐  │
│                                  ▲            │   Low-level     │  │
│                                  └────────────│   Controller    │  │
│                                               └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Figure 1: Governor-centered architecture. The Spiral-Time Governor evaluates
instability ΔΦ and enforces deterministic gating of LLM outputs before execution.
The feedback loop from the controller returns telemetry to the governor.
```

---

## 7 · Stability Statement (Discrete Lyapunov / ISS)

### 7.1 · Conservative Linear Envelope Model

Define **x(t) := (ϕ(t), χ(t))ᵀ ∈ ℝ²**. Under normalization/clamping:

$$x(t+1) \;=\; A\,x(t) \;+\; B\,\eta(t), \qquad \|\eta(t)\| \;\leq\; \bar{\eta}$$

Assume **ρ(A) < 1** (spectral radius strictly less than 1).

### 7.2 · Explicit Lyapunov Construction

Choose **Q ≻ 0** (e.g., Q = I). Then there exists unique **P ≻ 0** such that:

$$P \;-\; A^\top P A \;=\; Q$$

Let **V(x) = xᵀPx**.

---

> **Theorem 1** *(ISS-style boundedness via discrete Lyapunov equation.)*
> Assume ρ(A) < 1 and ‖η(t)‖ ≤ η̄ for all t. Then along trajectories:
>
> $$V(x(t+1)) - V(x(t)) \;\leq\; -\lambda_{\min}(Q)\,\|x(t)\|^2 \;+\; 2\,\|A^\top PB\|\,\|x(t)\|\,\|\eta(t)\| \;+\; \lambda_{\max}(B^\top PB)\,\|\eta(t)\|^2$$
>
> Hence there exist constants c₀, c₁ > 0 such that:
>
> $$\boxed{\|x(t)\| \;\leq\; c_0\,\rho(A)^t\,\|x(0)\| \;+\; c_1\,\bar{\eta}, \qquad \forall\, t \geq 0}$$

*Proof.* Expand V(x(t+1)) with x(t+1) = Ax(t) + Bη(t) and substitute P − AᵀPA = Q. Bound cross terms by Cauchy–Schwarz and eigenvalues. □

```
  Lyapunov decay illustration:
  ‖x(t)‖
     │
   ‖x(0)‖·c₀ ──┐
               │ ╲  exponential decay ρ(A)ᵗ
               │   ╲
               │     ╲_____
    c₁·η̄ ─────│──────────── ─ ─ ─ (bounded residual)
               │
               └──────────────────────────→ t
```

The stability perspective is conceptually aligned with barrier-function and Lyapunov-based safety formulations [1], although the present governor operates at the **semantic decision layer** rather than continuous torque control.

---

## 8 · Simulation Environment

All experiments reported in this study are conducted within a controlled synthetic stochastic testbed designed to isolate and evaluate the instability-gating behavior of the Spiral-Time Governor. The testbed is implemented as a deterministic JavaScript environment in which physical dynamics are replaced by a parametric noise model, enabling exact reproducibility across platforms without dependency on external physics simulation infrastructure.

> **Synthetic Testbed Disclaimer.** The primary quantitative results presented in this manuscript are obtained from the controlled synthetic environment described above. The testbed isolates the deterministic gating dynamics of the governor under structured stochastic perturbations. A complementary physics-grounded MuJoCo validation is provided in §8.2 as a minimal transfer experiment without parameter retuning.

Five experimental conditions are evaluated across three tasks of increasing difficulty:

| Task | Label | Noise Multiplier | Description |
|---|---|---|---|
| T1 | Climb | 1.00 | Discrete holds, friction variation, sensor dropout |
| T2 | Stairs | 1.15 | Irregular stairs, IMU bias bursts, occlusion windows |
| T3 | Gap | 1.30 | Gap crossing, external pushes, terrain class changes |

Each episode: **120 discrete time steps** · **N = 30 seeds** per condition · **90 episodes** per condition · **54,000 total simulated steps**.

### 8.1 · Statistical Analysis

All statistical analyses use **α = 0.05** (two-tailed).

- **Primary endpoint:** Hallucination rate H_T
- **Secondary endpoints:** Safety violations, success rate, action variance
- **Confidence intervals:** Bootstrap percentile method (B = 2,000 resamples, fixed LCG seed 77)
- **Pairwise tests:** Mann–Whitney U (normal approximation, two-tailed)
- **Multiple comparisons:** Holm step-down procedure (5 planned contrasts)
- **Effect sizes:** Rank-biserial correlation r = |Z|/√N and Cliff's δ

**Table 2:** Primary metrics (mean [95% CI]). N = 90 per condition.

| Condition | H_T ↓ | Violations ↓ | Success (%) ↑ | Action Var ↓ |
|---|---|---|---|---|
| Baseline LLM | 0.4595 [0.4491, 0.4702] | 13.24 [12.59, 13.96] | 34.4 [24.4, 44.4] | 0.0472 [0.0465, 0.0480] |
| LLM+RAG | 0.3679 [0.3585, 0.3775] | 12.56 [11.94, 13.18] | 38.9 [28.9, 48.9] | 0.0470 [0.0463, 0.0477] |
| **LLM+Governor** | **0.2200 [0.2139, 0.2266]** | **12.59 [11.82, 13.38]** | **41.1 [31.1, 51.1]** | **0.0474 [0.0466, 0.0481]** |
| Ablation A (δ=0) | 0.2422 [0.2350, 0.2497] | 13.79 [13.03, 14.49] | 30.0 [21.1, 38.9] | 0.0475 [0.0468, 0.0482] |
| Ablation B (always-exec) | 0.4369 [0.4270, 0.4473] | 24.68 [23.80, 25.57] | 0.0 [0.0, 0.0] | 0.0477 [0.0470, 0.0485] |

```
  H_T by condition (mean, lower bar = 95% CI):

  0.50 ┤
  0.45 ┤  ████  Baseline (0.4595)
  0.40 ┤  ████
  0.35 ┤  ████  ████  RAG (0.3679)
  0.30 ┤  ████  ████
  0.25 ┤  ████  ████              ████  Abl-A (0.2422)
  0.20 ┤  ████  ████  ████  ████  ████
       │              Gov   Abl-B
       │             (0.22)(0.44)
  0.00 ┴──────────────────────────────
        Base   RAG   Gov  Abl-B  Abl-A

  ↓ Lower is better. Governor achieves lowest H_T.
```

**Table 3:** Pairwise comparisons (Mann–Whitney U, Holm-corrected).

| Comparison | Endpoint | U | Z | Holm p | r | Significance |
|---|---|---|---|---|---|---|
| Gov vs. Baseline | H_T | 8100 | +11.587 | < 10⁻¹⁰ | 0.864 | *** |
| Gov vs. RAG | H_T | 8084 | +11.541 | < 10⁻¹⁰ | 0.860 | *** |
| Abl-A vs. Gov | H_T | 2663 | −3.970 | < 0.001 | 0.296 | *** |
| Abl-B vs. Gov | H_T | 0 | −11.587 | < 10⁻¹⁰ | 0.864 | *** |
| Gov vs. Baseline | Violations | 4473 | +1.210 | 0.226 | 0.090 | ns |

*\*\*\* p < 0.001 · ns = not significant*

**Table 4:** Per-step governor computational overhead.

| Component | Latency (ms) | CPU (%) | Mem (MB) | Energy (mJ) |
|---|---|---|---|---|
| Claim verification ΔI | 1.2 ± 0.2 | 2.1 | 4.2 | ~0.8 |
| Constraint checks ΔR | 0.9 ± 0.1 | 1.8 | 3.1 | ~0.6 |
| Contradiction score ΔC | 2.4 ± 0.4 | 3.5 | 8.6 | ~1.4 |
| Governor update (ϕ, χ, ΔΦ) | 0.3 ± 0.1 | 0.4 | 1.1 | ~0.2 |
| Logging / audit trail | 0.6 ± 0.1 | 0.9 | 2.3 | ~0.3 |
| **TOTAL** | **5.4 ± 0.7** | **8.7** | **19.3** | **~3.3** |

> 💡 Total overhead of **5.4 ms/step** is within real-time constraints at typical legged robot control frequencies (50–200 Hz → 5–20 ms budget per step).

### 8.2 · Embodied MuJoCo Validation (Companion PoC)

To evaluate whether the Spiral-Time Governor transfers from the synthetic testbed to a physics-grounded setting, we implemented a minimal embodied validation using the **dm_control quadruped domain** (MuJoCo backend).

The dm_control `"escape"` task was used as a physics-grounded proxy for the ANYmal-class terrain locomotion tasks described in this work. While not a full terrain benchmark, the task provides realistic contact dynamics, joint actuation, and proprioceptive observations under a deterministic physics engine.

```
  MuJoCo Validation Pipeline:

  ┌──────────────────────────────────────────────────────┐
  │  dm_control quadruped "escape" task (MuJoCo 3.x)    │
  │                                                      │
  │  Physics state sₜ ──▶ Oracle 𝒪(sₜ)                  │
  │     • torso_pos                                      │
  │     • torso_upright (quaternion w-component)         │
  │     • contact_flags [lf, rf, lh, rh feet]           │
  │     • n_contacts, feasible, terrain_class            │
  │                │                                     │
  │                ▼                                     │
  │  Mock LLM Agent (deterministic, seed-controlled)     │
  │     • hallucination_prob = 0.45 (all conditions)     │
  │     • sinusoidal gait + noise action proposals       │
  │                │                                     │
  │                ▼                                     │
  │  Spiral-Time Governor (params identical to v2.2)     │
  │     • ΔR, ΔI, ΔC from real physics state            │
  │     • Mode: EXECUTE / VERIFY / SAFE                  │
  └──────────────────────────────────────────────────────┘
```

**Protocol consistency.** All STG parameters (wR, wI, wC, α, β, γ, δ, τ₁, τ₂, ϕ₀) were kept **identical** to those used in the synthetic v2.2 evaluation. No retuning was performed. Experimental seeds (0–9) were used for primary reporting. Additional robustness seeds (40–49) were executed as validation checks and yielded consistent qualitative behavior (not shown). Statistical analysis followed the same bootstrap protocol (B = 2,000 resamples).

**Agent model.** A deterministic mock LLM agent was used to generate claims and action proposals with configurable hallucination probability. The governor operated strictly as an external gating layer without modifying the underlying action distribution.

**Quantitative outcome.** The governor condition reduces H_T from 0.41 to 0.24, corresponding to an approximate **relative reduction of 41%**, while preserving or slightly improving task success under contact-rich dynamics. Deterministic mode switching (EXECUTE / VERIFY / SAFE) was observed consistently across all runs.

**Table 5:** Embodied MuJoCo validation — mean and 95% bootstrap CI over seeds 0–9.

| Condition | H_T ↓ | 95% CI | Success (%) ↑ |
|---|---|---|---|
| Baseline LLM | 0.41 | [0.38, 0.44] | 36.0 |
| **LLM+Governor** | **0.24** | **[0.21, 0.27]** | **43.0** |

```
  MuJoCo H_T comparison (seeds 0–9):

  0.50 ┤
  0.45 ┤
  0.41 ┤  ████████████  Baseline  [0.38─────0.44]
  0.35 ┤  ████████████
  0.30 ┤  ████████████
  0.25 ┤  ████████████
  0.24 ┤  ████████████  ▓▓▓▓▓▓▓▓  Governor [0.21─────0.27]
  0.20 ┤               ▓▓▓▓▓▓▓▓
       │
  0.00 ┴──────────────────────────
              Baseline   Governor

  ≈ 41% relative reduction in hallucination rate.
  Non-overlapping CIs confirm significant separation.
```

The complete reproducible implementation, configuration files, seed lists, CI workflow, and analysis scripts are publicly available at:

> **[https://github.com/dfeen87/stg-embodied-poc](https://github.com/dfeen87/stg-embodied-poc)**

**Scope and limitations.** This PoC is intended as a **minimal physics-grounded transfer validation** rather than a full locomotion benchmark. The quadruped `"escape"` task serves as a contact-dynamics proxy. Custom terrain assets and hardware-level controllers are outside the scope of this study. The LLM component is mocked for determinism and reproducibility; integration with external API-based LLMs is left for future work.

---

## 9 · Limitations

The proposed Spiral-Time Governor provides deterministic mode switching and bounded instability under the assumptions stated in §7. However, several limitations should be noted.

**Conservative supervision.** The governor enforces threshold-based switching to reduce instability but does not guarantee semantic correctness of LLM outputs. It mitigates divergence rather than proving task validity.

**Threshold sensitivity.** Performance depends on the choice of instability thresholds τ₁ and τ₂. While fixed parameters and ablation studies reduce tuning bias, adaptive or theoretically derived thresholds remain future work.

**Simulator-to-hardware transfer.** The MuJoCo validation serves as a physics-grounded proxy. Real hardware deployment would require certified monitoring layers and redundancy checks replacing the simulator oracle.

**LLM abstraction.** The embodied experiments use a deterministic mock LLM for reproducibility. Integration with external API-based LLMs may introduce additional latency, stochasticity, and distribution shift.

---

## 10 · Conclusion

We presented a **deterministic Spiral-Time Governor** that suppresses hallucination-driven divergence in LLM-assisted climbing and walking robotics via a transparent instability functional ΔΦ, deterministic threshold gating, and discrete Lyapunov boundedness under bounded disturbances.

The proposed governance layer is model-agnostic, does not modify LLM weights, and provides auditable execution control through explicit instability thresholds and mode switching (EXECUTE / VERIFY / SAFE).

A **two-layer evaluation protocol** was established, consisting of a controlled synthetic stochastic testbed and a complementary MuJoCo-based physics-grounded transfer validation without parameter retuning. Reproducible reporting templates (effect sizes, Holm-adjusted significance testing, and compute/latency/power overhead analysis) are provided to support independent verification.

The framework offers a deterministic safety supervision mechanism suitable for long-horizon deployment of LLM-assisted legged robotic systems.

---

## Acknowledgements

Repository structure, implementation scaffolding, and documentation were developed with assistance from **Claude (Sonnet 4.6)** by Anthropic ([https://www.anthropic.com](https://www.anthropic.com)). All scientific content, mathematical formulations, experimental design, and statistical analysis originate from the authors.

---

## Funding

This research received no external funding. The work was conducted independently by the authors.

---

## Data Availability

The synthetic simulation code, parameter configurations, fixed seed lists, and statistical evaluation scripts are available in a version-controlled public repository:

> **[https://github.com/dfeen87/stg-embodied-poc](https://github.com/dfeen87/stg-embodied-poc)**

The repository contains:
1. Full simulation source code
2. Configuration files for all experimental conditions
3. Logged episode-level outputs (`.jsonl`)
4. Statistical evaluation and bootstrap scripts
5. Fixed seed lists ensuring deterministic reproducibility

The MuJoCo-based embodied validation environment is included as a companion implementation. All governor parameters are fixed and disclosed to enable exact replication of the reported results.

---

## AI Usage Statement

Large Language Models (LLMs) were used solely as **research objects** within the experimental framework described in this manuscript.

Generative AI tools were additionally used for language refinement and minor formatting assistance. All mathematical derivations, algorithmic formulations, experimental design, statistical analysis, and scientific conclusions were developed and verified by the authors.

*The authors take full responsibility for the content of this manuscript.*

---

## Conflict of Interest

The authors declare no conflict of interest.

---

## Author Contributions

**M.K.** is the originator of the Spiral-Time governance concept and primary author of the manuscript. He developed the complete theoretical framework, mathematical formalism, instability functional, stability proof, and overall system design.

**D.M.F.** was responsible for implementation of the simulation testbed, experimental execution, data generation, and statistical evaluation. He contributed to validation of the framework under controlled perturbations and provided technical feedback during manuscript refinement.

*All authors approved the final manuscript.*

---

## References

[1] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety-Critical Systems," *IEEE Transactions on Automatic Control*, vol. 62, no. 8, pp. 3861–3876, Aug. 2017. doi: [10.1109/TAC.2016.2638961](https://doi.org/10.1109/TAC.2016.2638961)

[2] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, P. Sermanet, N. Brown, T. Jackson, S. Levine, V. Vanhoucke, and K. Hausman, "Inner Monologue: Embodied Reasoning through Planning with Language Models," *arXiv preprint arXiv:2207.05608*, 2022. [https://arxiv.org/abs/2207.05608](https://arxiv.org/abs/2207.05608)

[3] A. Z. Ren, B. Govil, T.-Y. Yang, K. Narasimhan, and A. Majumdar, "Robots that Ask for Help: Uncertainty-Aligned LLM Planning," *arXiv preprint arXiv:2307.01928*, 2023. [https://arxiv.org/abs/2307.01928](https://arxiv.org/abs/2307.01928)

[4] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023. doi: [10.1145/3571730](https://doi.org/10.1145/3571730)

[5] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu, "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," *arXiv preprint arXiv:2311.05232*, 2023. [https://arxiv.org/abs/2311.05232](https://arxiv.org/abs/2311.05232)

[6] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language Agents with Iterative Design Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[7] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control Barrier Functions: Theory and Applications," in *Proc. IEEE 18th European Control Conference (ECC)*, Naples, Italy, 2019, pp. 3420–3431. doi: [10.23919/ECC.2019.8796030](https://doi.org/10.23919/ECC.2019.8796030)

[8] R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, "Safe Control with Learned Models: Optimality and Runtime Guarantees," *IEEE Transactions on Automatic Control*, 2023. doi: [10.1109/TAC.2023.3247173](https://doi.org/10.1109/TAC.2023.3247173)

[9] S. Gu et al., "Safe Multi-Agent Reinforcement Learning for Climbing Robots in Uncertain Environments," *Journal of Intelligent & Robotic Systems*, vol. 110, 2024.

[10] H. K. Khalil, *Nonlinear Systems*, 3rd ed. Upper Saddle River, NJ: Prentice Hall, 2002. ISBN: 978-0130673893.

[11] J.-J. E. Slotine and W. Li, *Applied Nonlinear Control*. Englewood Cliffs, NJ: Prentice Hall, 1991. ISBN: 978-0130408907.

[12] E. D. Sontag, "Input-to-State Stability: Basic Concepts and Results," in *Nonlinear Dynamics and Operational Control*, P. Nistri and G. Stefani, Eds. Berlin: Springer, 1989, pp. 163–220.
