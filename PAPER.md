# Deterministic Spiral-Time Governance for Hallucination Suppression in LLM-Controlled Climbing and Walking Robots

**Marcel Krüger¹ · Don Michael Feeney Jr.²**

¹ Independent Researcher, Germany

² Independent Researcher, USA

Corresponding author: marcelkrueger092@gmail.com

ORCID (M.K.): 0009-0002-5709-9729 

ORCID (D.M.F.): 0009-0003-1350-4160

*Journal of Climbing and Walking Robots · 2026 · Vol. XX(XX) · pp. 1–XX*
DOI: *(assigned by journal)*

---

## Abstract

Large Language Models (LLMs) are increasingly integrated into robotic autonomy stacks for semantic planning and high-level decision support. In climbing and walking robots, long-horizon deployment is constrained by hallucination drift, retrieval instability, and non-deterministic reasoning variance, which can propagate into physically unsafe locomotion behavior.

We introduce a deterministic external governance layer based on a Spiral-Time operator formalism. Interaction history is embedded into a triadic state ψ(t) = t + iϕ(t) + jχ(t), where ϕ(t) encodes contextual coherence and χ(t) = ∂ₜϕ(t) captures temporal torsion associated with abrupt divergence. A scalar instability functional ΔΦ(t) deterministically regulates memory writes, retrieval widening, verification mode, and safe fallback switching.

Crucially, we define hallucination operationally in robotics as **falsifiable inconsistency** between LLM-issued claims/commands and a ground-truth oracle derived from simulator state. We provide (i) a formal supervisor model, (ii) a deterministic gating algorithm, and (iii) a discrete Lyapunov-based boundedness/ISS argument for the governor state under bounded measurement noise. A reproducible simulation protocol is specified for climbing and walking tasks with controlled perturbations and statistical evaluation (seeds, confidence intervals, ablation tests).

In addition to the synthetic stochastic evaluation, we provide a **minimal physics-grounded transfer validation** using a MuJoCo-based quadruped environment. All governor parameters are kept identical to the synthetic setting without retuning. The embodied validation reproduces the qualitative reduction in unsafe actions and deterministic mode switching behavior under contact-rich dynamics, while leaving the underlying hallucination statistics unchanged.

The framework is model-agnostic and does not modify LLM weights, offering auditability and predictable behavior suitable for safety-sensitive legged robot deployments.

**Keywords:** legged robotics · LLM-based control · hallucination detection · deterministic safety supervision · MuJoCo validation

---

## 1 · Introduction

Climbing and walking robots operate under strong contact constraints, terrain discontinuities, and safety-critical failure modes (falls, self-collision, uncontrolled impacts). The stability perspective adopted here is conceptually aligned with barrier-function and Lyapunov-based safety formulations in control theory [1], although the present governor operates at the **semantic decision layer** rather than continuous torque control.

Recent robotics research increasingly integrates LLMs as high-level planners, semantic interfaces, and vision-language-action controllers. While such models enable flexible task abstraction and semantic reasoning, their generative nature introduces instability risks: hallucinated environment claims, inconsistent tool outputs, and non-deterministic subgoal sequences can propagate into physically unsafe locomotion behavior — especially under partial observability and long-horizon deployment.

Existing mitigation strategies typically rely on confidence scoring, retrieval augmentation, or post-hoc verification. However, these approaches remain **probabilistic** and do not provide deterministic guarantees on mode switching or bounded supervisory dynamics.

This paper targets a specific failure class: hallucinated assertions or action recommendations that are inconsistent with verified robot state and constraints. We propose a **deterministic Spiral-Time Governor** that wraps a black-box LLM and enforces auditable threshold-based mode switching.

<!-- FIGURE: Mode Switching Overview -->
<svg width="100%" viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- EXECUTE box -->
  <rect x="40" y="30" width="170" height="90" rx="10" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.2"/>
  <text x="125" y="60" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#1b5e20">EXECUTE</text>
  <text x="125" y="80" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#2e7d32">Normal operation</text>
  <text x="125" y="97" text-anchor="middle" font-family="monospace" font-size="10" fill="#388e3c">ΔΦ(t) &lt; τ₁</text>
  <text x="125" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#388e3c">τ₁ = 0.25</text>
  <!-- VERIFY box -->
  <rect x="255" y="30" width="170" height="90" rx="10" fill="#fff8e1" stroke="#f57f17" stroke-width="1.2"/>
  <text x="340" y="60" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#e65100">VERIFY</text>
  <text x="340" y="80" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#f57f17">Cross-check &amp; validate</text>
  <text x="340" y="97" text-anchor="middle" font-family="monospace" font-size="10" fill="#fb8c00">τ₁ ≤ ΔΦ(t) &lt; τ₂</text>
  <text x="340" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#fb8c00">τ₁=0.25, τ₂=0.55</text>
  <!-- SAFE box -->
  <rect x="470" y="30" width="170" height="90" rx="10" fill="#ffebee" stroke="#c62828" stroke-width="1.2"/>
  <text x="555" y="60" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#b71c1c">SAFE</text>
  <text x="555" y="80" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#c62828">Freeze + fallback</text>
  <text x="555" y="97" text-anchor="middle" font-family="monospace" font-size="10" fill="#e53935">ΔΦ(t) ≥ τ₂</text>
  <text x="555" y="112" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#e53935">τ₂ = 0.55</text>
  <!-- arrows -->
  <line x1="210" y1="75" x2="253" y2="75" stroke="#888" stroke-width="1.2" marker-end="url(#arr1)"/>
  <line x1="425" y1="75" x2="468" y2="75" stroke="#888" stroke-width="1.2" marker-end="url(#arr1)"/>
  <!-- ΔΦ axis label -->
  <text x="340" y="148" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">ΔΦ(t) increases →  instability grows →  more conservative mode</text>
</svg>

*Figure 1: The three deterministic operating modes of the Spiral-Time Governor, triggered by the instability functional ΔΦ(t) against fixed thresholds τ₁ = 0.25 and τ₂ = 0.55.*

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

<!-- FIGURE: Robot Autonomy Stack -->
<svg width="100%" viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- outer container -->
  <rect x="30" y="15" width="620" height="265" rx="14" fill="none" stroke="#b0bec5" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="50" y="36" font-family="sans-serif" font-size="11" fill="#78909c">Physical Robot / Simulator</text>
  <!-- Robot state box -->
  <rect x="55" y="50" width="160" height="56" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1"/>
  <text x="135" y="73" text-anchor="middle" font-family="monospace" font-size="12" font-weight="600" fill="#0d47a1">sₜ ∈ 𝒮</text>
  <text x="135" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Robot state</text>
  <!-- Measured outputs -->
  <rect x="255" y="50" width="160" height="56" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1"/>
  <text x="335" y="73" text-anchor="middle" font-family="monospace" font-size="12" font-weight="600" fill="#0d47a1">yₜ ∈ 𝒴</text>
  <text x="335" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Measured outputs</text>
  <!-- Low-level controller -->
  <rect x="455" y="50" width="160" height="56" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1"/>
  <text x="535" y="73" text-anchor="middle" font-family="monospace" font-size="12" font-weight="600" fill="#0d47a1">uₜ ∈ 𝒰</text>
  <text x="535" y="92" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Low-level controller</text>
  <!-- arrows top row -->
  <line x1="215" y1="78" x2="253" y2="78" stroke="#1565c0" stroke-width="1.2" marker-end="url(#arr2)"/>
  <line x1="415" y1="78" x2="453" y2="78" stroke="#1565c0" stroke-width="1.2" marker-end="url(#arr2)"/>
  <!-- LLM Module -->
  <rect x="155" y="165" width="170" height="60" rx="8" fill="#f3e5f5" stroke="#6a1b9a" stroke-width="1"/>
  <text x="240" y="188" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#4a148c">LLM Module</text>
  <text x="240" y="206" text-anchor="middle" font-family="monospace" font-size="10" fill="#6a1b9a">â_t claims, subgoals</text>
  <text x="240" y="220" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6a1b9a">action proposals</text>
  <!-- Governor -->
  <rect x="355" y="165" width="180" height="60" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="1.5"/>
  <text x="445" y="188" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#bf360c">Governor 𝒢</text>
  <text x="445" y="206" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#e64a19">Deterministic supervisor</text>
  <text x="445" y="220" text-anchor="middle" font-family="monospace" font-size="10" fill="#e64a19">admissible?</text>
  <!-- arrow: yₜ down to LLM -->
  <line x1="335" y1="106" x2="285" y2="163" stroke="#888" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arr2)"/>
  <!-- arrow: LLM to Governor -->
  <line x1="325" y1="195" x2="353" y2="195" stroke="#6a1b9a" stroke-width="1.2" marker-end="url(#arr2)"/>
  <!-- arrow: Governor to uₜ -->
  <line x1="535" y1="163" x2="535" y2="108" stroke="#e65100" stroke-width="1.2" marker-end="url(#arr2)"/>
  <line x1="445" y1="163" x2="480" y2="130" stroke="#e65100" stroke-width="1" stroke-dasharray="3 3" marker-end="url(#arr2)"/>
  <!-- label -->
  <text x="340" y="280" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 2: Robot autonomy stack with LLM module and deterministic Governor 𝒢.</text>
</svg>

### 2.2 · Oracle-Based Hallucination Definition

In simulation we have a ground-truth oracle 𝒪(sₜ) providing verifiable predicates (pose, contact set, terrain class, feasibility flags). Let the LLM output at time t include claims Kₜ = {kₜ,₁, …, kₜ,mₜ} and proposed action â_t.

Define a verification map **ver(kₜ,ⱼ) ∈ {0, 1}** — equals 1 iff kₜ,ⱼ is consistent with 𝒪(sₜ) and certified checks.

**Hallucination rate over horizon T:**

$$H_T := \frac{1}{\sum_{t=1}^{T} m_t} \sum_{t=1}^{T}\sum_{j=1}^{m_t} \bigl(1 - \text{ver}(k_{t,j})\bigr)$$

Thus hallucination is defined as a **testable inconsistency** — not a probabilistic confidence estimate.

---

## 3 · Spiral-Time Governor: State, Instability, and Gating

### 3.1 · Triadic Spiral-Time Embedding

We define the triadic state: **ψ(t) = t + iϕ(t) + jχ(t)**, where χ(t) := ϕ(t) − ϕ(t−1).

Here ϕ(t) ∈ [0, 1] is a deterministic coherence score and χ(t) captures abrupt coherence changes (temporal torsion).

<!-- FIGURE: Spiral-Time State Space -->
<svg width="100%" viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Background panel -->
  <rect x="30" y="20" width="380" height="255" rx="12" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>
  <!-- Axes -->
  <!-- t axis (diagonal = depth illusion) -->
  <line x1="120" y1="230" x2="60" y2="270" stroke="#555" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="42" y="280" font-family="serif" font-size="13" font-style="italic" fill="#333">t</text>
  <!-- iϕ axis (horizontal right) -->
  <line x1="120" y1="230" x2="340" y2="230" stroke="#1565c0" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="348" y="234" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">iϕ(t)</text>
  <!-- jχ axis (vertical up) -->
  <line x1="120" y1="230" x2="120" y2="50" stroke="#6a1b9a" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="96" y="44" font-family="serif" font-size="13" font-style="italic" fill="#6a1b9a">jχ(t)</text>
  <!-- Origin dot -->
  <circle cx="120" cy="230" r="3" fill="#333"/>
  <text x="100" y="248" font-family="sans-serif" font-size="10" fill="#555">origin</text>
  <!-- Sample ψ(t) point -->
  <circle cx="248" cy="118" r="7" fill="#e65100" stroke="#fff" stroke-width="2"/>
  <text x="260" y="113" font-family="serif" font-size="13" font-style="italic" fill="#e65100">ψ(t)</text>
  <!-- dashed projection lines -->
  <line x1="248" y1="118" x2="248" y2="230" stroke="#1565c0" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="120" y1="118" x2="248" y2="118" stroke="#6a1b9a" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="120" y1="230" x2="248" y2="118" stroke="#e65100" stroke-width="1.5" stroke-dasharray="3 2"/>
  <!-- coordinate labels -->
  <text x="184" y="245" font-family="monospace" font-size="10" fill="#1565c0">ϕ(t)</text>
  <text x="68" y="175" font-family="monospace" font-size="10" fill="#6a1b9a">χ(t)</text>
  <!-- formula panel -->
  <rect x="430" y="30" width="225" height="235" rx="10" fill="#fff8e1" stroke="#f9a825" stroke-width="1"/>
  <text x="542" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#e65100">Triadic State</text>
  <text x="542" y="82" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">ψ(t) = t + iϕ(t) + jχ(t)</text>
  <line x1="448" y1="95" x2="637" y2="95" stroke="#f9a825" stroke-width="0.8"/>
  <text x="448" y="116" font-family="sans-serif" font-size="11" font-weight="600" fill="#333">Components:</text>
  <text x="448" y="136" font-family="monospace" font-size="10" fill="#333">t      — real time</text>
  <text x="448" y="156" font-family="monospace" font-size="10" fill="#1565c0">iϕ(t) — coherence</text>
  <text x="448" y="176" font-family="monospace" font-size="10" fill="#6a1b9a">jχ(t) — torsion</text>
  <line x1="448" y1="190" x2="637" y2="190" stroke="#f9a825" stroke-width="0.8"/>
  <text x="448" y="210" font-family="sans-serif" font-size="11" font-weight="600" fill="#333">Torsion:</text>
  <text x="448" y="230" font-family="serif" font-size="12" font-style="italic" fill="#6a1b9a">χ(t) = ϕ(t) − ϕ(t−1)</text>
  <text x="448" y="252" font-family="sans-serif" font-size="10" fill="#555">captures abrupt divergence</text>
  <!-- caption -->
  <text x="340" y="288" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 3: Triadic Spiral-Time state space. ψ(t) traces a path as coherence and torsion evolve.</text>
</svg>

### 3.2 · Deterministic Coherence Components

We use deterministic deviations inspired by the Structure–Information–Coherence (S–I–C) framework:

| Component | Symbol | Range | Captures |
|---|---|---|---|
| Structure deviation | ΔR(t) | [0, 1] | Constraint/plan mismatch, inadmissible tools, feasibility failures |
| Information deviation | ΔI(t) | [0, 1] | Claim mismatch against oracle/telemetry |
| Coherence deviation | ΔC(t) | [0, 1] | Contradiction score vs. verified memory window |

**Information deviation:** ΔI(t) := (1/mₜ) · Σⱼ (1 − ver(kₜ,ⱼ))

**Coherence score:** ϕ(t) := 1 − (wR·ΔR(t) + wI·ΔI(t) + wC·ΔC(t)),   where wR + wI + wC = 1

### 3.3 · Instability Functional

$$\boxed{\Delta\Phi(t) := \alpha\,\Delta R(t) + \beta\,\Delta I(t) + \gamma\,\Delta C(t) + \delta\,|\chi(t)|, \quad \alpha+\beta+\gamma+\delta = 1}$$

<!-- FIGURE: Instability Functional Decomposition -->
<svg width="100%" viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- ΔR component -->
  <rect x="40" y="30" width="118" height="64" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1"/>
  <text x="99" y="54" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#1b5e20">ΔR(t)</text>
  <text x="99" y="72" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e7d32">Structure</text>
  <text x="99" y="86" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#388e3c">α = 0.25</text>
  <!-- ΔI component -->
  <rect x="185" y="30" width="118" height="64" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1"/>
  <text x="244" y="54" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#0d47a1">ΔI(t)</text>
  <text x="244" y="72" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Information</text>
  <text x="244" y="86" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1976d2">β = 0.35</text>
  <!-- ΔC component -->
  <rect x="330" y="30" width="118" height="64" rx="8" fill="#f3e5f5" stroke="#6a1b9a" stroke-width="1"/>
  <text x="389" y="54" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#4a148c">ΔC(t)</text>
  <text x="389" y="72" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6a1b9a">Coherence</text>
  <text x="389" y="86" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#7b1fa2">γ = 0.25</text>
  <!-- |χ(t)| component -->
  <rect x="475" y="30" width="118" height="64" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="1"/>
  <text x="534" y="54" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#bf360c">|χ(t)|</text>
  <text x="534" y="72" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#e65100">Torsion</text>
  <text x="534" y="86" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#f57c00">δ = 0.15</text>
  <!-- arrows down to ΔΦ -->
  <line x1="99"  y1="94" x2="310" y2="158" stroke="#2e7d32" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="244" y1="94" x2="320" y2="158" stroke="#1565c0" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="389" y1="94" x2="340" y2="158" stroke="#6a1b9a" stroke-width="1" marker-end="url(#arr4)"/>
  <line x1="534" y1="94" x2="360" y2="158" stroke="#e65100" stroke-width="1" marker-end="url(#arr4)"/>
  <!-- ΔΦ output box -->
  <rect x="265" y="158" width="150" height="44" rx="8" fill="#212121" stroke="#424242" stroke-width="1.5"/>
  <text x="340" y="177" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#fff">ΔΦ(t)</text>
  <text x="340" y="194" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#bdbdbd">instability scalar</text>
  <text x="340" y="214" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#777">Figure 4: ΔΦ(t) is a weighted sum of four deterministic deviation signals.</text>
</svg>

### 3.4 · Deterministic Mode Switching

Thresholds 0 < τ₁ < τ₂ < 1 define:

| Condition | Mode |
|---|---|
| ΔΦ(t) < τ₁ | **EXECUTE** |
| τ₁ ≤ ΔΦ(t) < τ₂ | **VERIFY** |
| ΔΦ(t) ≥ τ₂ | **SAFE** |



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
  7:  if ΔΦ(t) < τ₁  →  M(t) ← EXECUTE ; allow â_t if constraints pass
  8:  elif ΔΦ(t) < τ₂ →  M(t) ← VERIFY  ; cross-check; block irreversible
  9:  else            →  M(t) ← SAFE    ; freeze writes; trigger fallback
 10:  Log (t, ϕ(t), χ(t), ΔΦ(t), M(t))  →  immutable audit trail
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

<!-- FIGURE: System Architecture -->
<svg width="100%" viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Telemetry box -->
  <rect x="40" y="50" width="150" height="80" rx="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.2"/>
  <text x="115" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#0d47a1">Telemetry</text>
  <text x="115" y="97" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Perception / SLAM</text>
  <text x="115" y="113" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">Contacts / IMU</text>
  <!-- Spiral-Time Governor (central) -->
  <rect x="240" y="30" width="200" height="130" rx="12" fill="#fff8e1" stroke="#f9a825" stroke-width="2"/>
  <text x="340" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#e65100">Spiral-Time Governor</text>
  <line x1="258" y1="68" x2="422" y2="68" stroke="#f9a825" stroke-width="0.8"/>
  <text x="340" y="87" text-anchor="middle" font-family="monospace" font-size="11" fill="#555">ϕ(t)  coherence score</text>
  <text x="340" y="103" text-anchor="middle" font-family="monospace" font-size="11" fill="#555">χ(t)  torsion</text>
  <text x="340" y="119" text-anchor="middle" font-family="monospace" font-size="11" fill="#555">ΔΦ(t) instability</text>
  <rect x="262" y="131" width="156" height="20" rx="5" fill="#212121"/>
  <text x="340" y="145" text-anchor="middle" font-family="monospace" font-size="10" fill="#fff">EXECUTE · VERIFY · SAFE</text>
  <!-- LLM Agent -->
  <rect x="490" y="50" width="150" height="80" rx="10" fill="#f3e5f5" stroke="#6a1b9a" stroke-width="1.2"/>
  <text x="565" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#4a148c">LLM Agent</text>
  <text x="565" y="97" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6a1b9a">black-box</text>
  <text x="565" y="113" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6a1b9a">claims + proposals</text>
  <!-- Planner/Supervisor -->
  <rect x="340" y="225" width="160" height="60" rx="10" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1.2"/>
  <text x="420" y="250" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1b5e20">Planner / Supervisor</text>
  <text x="420" y="272" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e7d32">goal refinement</text>
  <!-- Low-level Controller -->
  <rect x="340" y="305" width="160" height="44" rx="10" fill="#ffebee" stroke="#c62828" stroke-width="1.2"/>
  <text x="420" y="326" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#b71c1c">Low-level Controller</text>
  <text x="420" y="342" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#c62828">PID / torque</text>
  <!-- Arrows -->
  <!-- Telemetry → Governor -->
  <line x1="190" y1="90" x2="238" y2="90" stroke="#1565c0" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Governor ↔ LLM -->
  <line x1="440" y1="80" x2="488" y2="80" stroke="#f9a825" stroke-width="1.5" marker-end="url(#arr5)"/>
  <line x1="490" y1="100" x2="442" y2="100" stroke="#6a1b9a" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arr5)"/>
  <!-- Governor → Planner -->
  <line x1="340" y1="160" x2="390" y2="223" stroke="#f9a825" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Planner → Controller -->
  <line x1="420" y1="285" x2="420" y2="303" stroke="#2e7d32" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Controller feedback → Telemetry (curved) -->
  <path d="M340 327 Q180 330 115 132" fill="none" stroke="#c62828" stroke-width="1" stroke-dasharray="5 3" marker-end="url(#arr5)"/>
  <text x="190" y="340" font-family="sans-serif" font-size="10" fill="#c62828">feedback</text>
  <!-- Legend labels -->
  <text x="340" y="355" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 5: Governor-centered architecture. The STG gates all LLM outputs before execution.</text>
</svg>

---

## 7 · Stability Statement (Discrete Lyapunov / ISS)

### 7.1 · Conservative Linear Envelope Model

Define **x(t) := (ϕ(t), χ(t))ᵀ ∈ ℝ²**. Under normalization/clamping:

$$x(t+1) = A\,x(t) + B\,\eta(t), \qquad \|\eta(t)\| \leq \bar{\eta}$$

Assume **ρ(A) < 1** (spectral radius strictly less than 1).

### 7.2 · Explicit Lyapunov Construction

Choose **Q ≻ 0** (e.g., Q = I). Then there exists unique **P ≻ 0** such that: **P − AᵀPA = Q**.

Let **V(x) = xᵀPx**.

> **Theorem 1** *(ISS-style boundedness via discrete Lyapunov equation.)*
> Assume ρ(A) < 1 and ‖η(t)‖ ≤ η̄ for all t. Then there exist constants c₀, c₁ > 0 such that:
>
> $$\boxed{\|x(t)\| \leq c_0\,\rho(A)^t\,\|x(0)\| + c_1\,\bar{\eta}, \qquad \forall\, t \geq 0}$$

*Proof.* Expand V(x(t+1)) with x(t+1) = Ax(t) + Bη(t) and substitute P − AᵀPA = Q. Bound cross terms by Cauchy–Schwarz and eigenvalues. □

<!-- FIGURE: Lyapunov Stability Decay -->
<svg width="100%" viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr6" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Plot area -->
  <rect x="60" y="20" width="500" height="175" rx="8" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>
  <!-- Axes -->
  <line x1="80" y1="180" x2="540" y2="180" stroke="#333" stroke-width="1.5" marker-end="url(#arr6)"/>
  <line x1="80" y1="180" x2="80" y2="30"  stroke="#333" stroke-width="1.5" marker-end="url(#arr6)"/>
  <text x="548" y="184" font-family="serif" font-size="13" font-style="italic" fill="#333">t</text>
  <text x="70"  y="26"  font-family="serif" font-size="13" font-style="italic" fill="#333">‖x‖</text>
  <!-- y-axis ticks -->
  <line x1="76" y1="50"  x2="84" y2="50"  stroke="#555" stroke-width="1"/>
  <line x1="76" y1="120" x2="84" y2="120" stroke="#555" stroke-width="1"/>
  <line x1="76" y1="155" x2="84" y2="155" stroke="#555" stroke-width="1"/>
  <text x="58" y="54"  font-family="monospace" font-size="10" fill="#555" text-anchor="end">‖x(0)‖·c₀</text>
  <text x="58" y="124" font-family="monospace" font-size="10" fill="#555" text-anchor="end">mid</text>
  <text x="58" y="159" font-family="monospace" font-size="10" fill="#c62828" text-anchor="end">c₁·η̄</text>
  <!-- Exponential decay curve -->
  <polyline points="80,50 120,72 160,92 200,108 240,120 280,130 320,138 360,144 400,149 440,153 480,156 520,157" fill="none" stroke="#1565c0" stroke-width="2.2"/>
  <!-- Residual bound line -->
  <line x1="80" y1="155" x2="530" y2="155" stroke="#c62828" stroke-width="1.5" stroke-dasharray="6 4"/>
  <!-- Shaded safe zone -->
  <rect x="80" y="155" width="450" height="25" fill="#ffebee" opacity="0.5"/>
  <!-- Annotations -->
  <text x="210" y="90" font-family="sans-serif" font-size="11" fill="#1565c0">c₀ · ρ(A)ᵗ · ‖x(0)‖</text>
  <text x="210" y="104" font-family="sans-serif" font-size="10" fill="#1565c0">exponential decay</text>
  <text x="380" y="150" font-family="sans-serif" font-size="10" fill="#c62828">c₁·η̄  (bounded residual)</text>
  <!-- Description -->
  <text x="310" y="218" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 6: ISS-style decay of governor state ‖x(t)‖. The system converges to a ball of radius c₁η̄.</text>
</svg>

---

## 8 · Simulation Environment

All experiments are conducted within a controlled synthetic stochastic testbed. The testbed is implemented as a deterministic JavaScript environment in which physical dynamics are replaced by a parametric noise model, enabling exact reproducibility.

> **Synthetic Testbed Disclaimer.** Primary quantitative results come from the controlled synthetic environment. A complementary MuJoCo validation is provided in §8.2 as a minimal transfer experiment without parameter retuning.

Five experimental conditions (Baseline LLM, LLM+RAG, LLM+Governor, Ablation A with δ = 0, and Ablation B always-execute) are evaluated across three tasks (T1: Climb, T2: Stairs, T3: Gap). Each episode consists of 120 discrete time steps. For each condition and task, N = 30 independent random seeds are used, yielding 90 episodes per condition and 54,000 total simulated steps.

### 8.1 · Statistical Analysis

All analyses use **α = 0.05** (two-tailed). CI: bootstrap percentile (B = 2,000, seed 77). Tests: Mann–Whitney U, Holm step-down. Effect sizes: rank-biserial r and Cliff's δ.

**Table 2:** Primary metrics (mean [95% CI]). N = 90 per condition.

| Condition | H_T ↓ | Violations ↓ | Success (%) ↑ | Action Var ↓ |
|---|---|---|---|---|
| Baseline LLM | 0.4595 [0.4491, 0.4702] | 13.24 [12.59, 13.96] | 34.4 [24.4, 44.4] | 0.0472 [0.0465, 0.0480] |
| LLM+RAG | 0.3679 [0.3585, 0.3775] | 12.56 [11.94, 13.18] | 38.9 [28.9, 48.9] | 0.0470 [0.0463, 0.0477] |
| **LLM+Governor** | **0.2200 [0.2139, 0.2266]** | **12.59 [11.82, 13.38]** | **41.1 [31.1, 51.1]** | **0.0474 [0.0466, 0.0481]** |
| Ablation A (δ=0) | 0.2422 [0.2350, 0.2497] | 13.79 [13.03, 14.49] | 30.0 [21.1, 38.9] | 0.0475 [0.0468, 0.0482] |
| Ablation B (always-exec) | 0.4369 [0.4270, 0.4473] | 24.68 [23.80, 25.57] | 0.0 [0.0, 0.0] | 0.0477 [0.0470, 0.0485] |

<!-- FIGURE: H_T Bar Chart -->
<svg width="100%" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
  <!-- Y-axis -->
  <line x1="80" y1="20" x2="80" y2="200" stroke="#333" stroke-width="1.5"/>
  <line x1="80" y1="200" x2="640" y2="200" stroke="#333" stroke-width="1.5"/>
  <!-- Y gridlines and labels -->
  <line x1="78" y1="200" x2="82" y2="200" stroke="#333" stroke-width="1.2"/>
  <line x1="78" y1="163" x2="82" y2="163" stroke="#333" stroke-width="1"/>
  <line x1="78" y1="126" x2="82" y2="126" stroke="#333" stroke-width="1"/>
  <line x1="78" y1="89"  x2="82" y2="89"  stroke="#333" stroke-width="1"/>
  <line x1="78" y1="52"  x2="82" y2="52"  stroke="#333" stroke-width="1"/>
  <text x="74" y="204" text-anchor="end" font-family="monospace" font-size="10" fill="#555">0.00</text>
  <text x="74" y="167" text-anchor="end" font-family="monospace" font-size="10" fill="#555">0.10</text>
  <text x="74" y="130" text-anchor="end" font-family="monospace" font-size="10" fill="#555">0.20</text>
  <text x="74" y="93"  text-anchor="end" font-family="monospace" font-size="10" fill="#555">0.30</text>
  <text x="74" y="56"  text-anchor="end" font-family="monospace" font-size="10" fill="#555">0.40</text>
  <!-- subtle gridlines -->
  <line x1="80" y1="163" x2="640" y2="163" stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="126" x2="640" y2="126" stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="89"  x2="640" y2="89"  stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="52"  x2="640" y2="52"  stroke="#eee" stroke-width="1"/>
  <!-- Bars: H_T values scaled: 0.50 → y=20, 0.00 → y=200. scale = 360px per unit -->
  <!-- Baseline: 0.4595 → height=165.4, top=34.6 -->
  <rect x="105" y="35" width="70" height="165" fill="#ef5350"/>
  <text x="140" y="30" text-anchor="middle" font-family="monospace" font-size="10" fill="#c62828">0.460</text>
  <!-- RAG: 0.3679 → height=132.4, top=67.6 -->
  <rect x="210" y="68" width="70" height="132" fill="#ff8f00"/>
  <text x="245" y="63" text-anchor="middle" font-family="monospace" font-size="10" fill="#e65100">0.368</text>
  <!-- Governor: 0.2200 → height=79.2, top=120.8 -->
  <rect x="315" y="121" width="70" height="79" fill="#43a047"/>
  <text x="350" y="116" text-anchor="middle" font-family="monospace" font-size="10" fill="#1b5e20" font-weight="700">0.220 ★</text>
  <!-- Ablation A: 0.2422 → height=87.2, top=112.8 -->
  <rect x="420" y="113" width="70" height="87" fill="#7e57c2"/>
  <text x="455" y="108" text-anchor="middle" font-family="monospace" font-size="10" fill="#4527a0">0.242</text>
  <!-- Ablation B: 0.4369 → height=157.3, top=42.7 -->
  <rect x="525" y="43" width="70" height="157" fill="#90a4ae"/>
  <text x="560" y="38" text-anchor="middle" font-family="monospace" font-size="10" fill="#455a64">0.437</text>
  <!-- X axis labels -->
  <text x="140" y="218" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">Baseline</text>
  <text x="245" y="218" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">RAG</text>
  <text x="350" y="218" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1b5e20" font-weight="700">Governor</text>
  <text x="455" y="218" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">Ablation A</text>
  <text x="560" y="218" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">Ablation B</text>
  <!-- Y axis label -->
  <text x="32" y="115" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555" transform="rotate(-90,32,115)">Hallucination Rate H_T ↓</text>
  <!-- Caption -->
  <text x="340" y="244" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 7: H_T by condition (lower is better). LLM+Governor achieves the lowest hallucination rate.</text>
</svg>

**Table 3:** Pairwise comparisons (Mann–Whitney U, Holm-corrected).

| Comparison | Endpoint | U | Z | Holm p | r | Sig. |
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

> Total overhead of **~5.4 ms/step** is within real-time constraints at typical legged robot control frequencies (50–200 Hz → 5–20 ms budget per step).

### 8.2 · Embodied MuJoCo Validation (Companion PoC)

To evaluate whether the Spiral-Time Governor transfers to a physics-grounded setting, we implemented a minimal embodied validation using the **dm_control quadruped domain** (MuJoCo backend). The `"escape"` task provides realistic contact dynamics, joint actuation, and proprioceptive observations.

**Protocol consistency.** All STG parameters (wR, wI, wC, α, β, γ, δ, τ₁, τ₂, ϕ₀) were kept identical to those used in the synthetic evaluation. No parameter retuning was performed for the embodied setting. Experimental seeds (0–9) were used for primary reporting, with additional validation seeds (40–49) confirming consistent qualitative behavior. Statistical analysis follows the bootstrap protocol defined in Section 8.1 (B = 2000 resamples).

**Agent model.** A deterministic mock LLM agent was used to generate claims and action proposals with a controlled hallucination probability. The STG was applied strictly as an external supervisory layer. Importantly, the governor does not modify the underlying generative process or action distribution of the LLM, but operates solely at the level of execution gating.

**Quantitative outcome.** After correcting the uprightness metric, identical hallucination rates are observed for both conditions (H_T = 0.65), indicating that the governor does not alter oracle–claim consistency or the statistical occurrence of hallucinated outputs. The primary observed effect is a substantial reduction in safety violations, decreasing from 11.4 to 4.7 violations per episode (approximately 59% reduction), with non-overlapping 95% bootstrap confidence intervals indicating statistical significance. Deterministic mode switching (EXECUTE / VERIFY / SAFE) was consistently observed across all runs. The fraction of EXECUTE mode decreased from 100% in the baseline condition to 30% under the governor, demonstrating active constraint-aware filtering while maintaining bounded computational overhead.

**Table 5:** Embodied MuJoCo validation — mean and 95% bootstrap CI over seeds 0–9.

| Condition | H_T | 95% CI | Violations ↓ | 95% CI | EXECUTE (%) |
|---|---|---|---|---|---|
| Baseline LLM | 0.65 | [0.47, 0.83] | 11.4 | [9.7, 13.1] | 100 |
| **LLM+Governor** | **0.65** | **[0.47, 0.83]** | **4.7** | **[3.0, 6.5]** | **30** |

<!-- FIGURE: MuJoCo Violations + Mode Split -->
<svg width="100%" viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg">
  <!-- LEFT PANEL: Violations comparison -->
  <rect x="30" y="15" width="290" height="220" rx="10" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>
  <text x="175" y="38" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#333">Unsafe-Action Violations</text>
  <!-- Y axis -->
  <line x1="70" y1="55" x2="70" y2="195" stroke="#555" stroke-width="1.2"/>
  <line x1="70" y1="195" x2="300" y2="195" stroke="#555" stroke-width="1.2"/>
  <!-- Y labels -->
  <text x="64" y="59"  text-anchor="end" font-family="monospace" font-size="9" fill="#555">12</text>
  <text x="64" y="105" text-anchor="end" font-family="monospace" font-size="9" fill="#555">8</text>
  <text x="64" y="150" text-anchor="end" font-family="monospace" font-size="9" fill="#555">4</text>
  <text x="64" y="196" text-anchor="end" font-family="monospace" font-size="9" fill="#555">0</text>
  <!-- gridlines -->
  <line x1="70" y1="59"  x2="300" y2="59"  stroke="#eee" stroke-width="1"/>
  <line x1="70" y1="105" x2="300" y2="105" stroke="#eee" stroke-width="1"/>
  <line x1="70" y1="150" x2="300" y2="150" stroke="#eee" stroke-width="1"/>
  <!-- Baseline bar: 11.4 violations, max ~12=55, scale=(195-55)/12=11.67 -->
  <rect x="105" y="62" width="65" height="133" fill="#ef5350"/>
  <!-- CI whiskers: 9.7–13.1 -->
  <line x1="137" y1="47" x2="137" y2="80" stroke="#c62828" stroke-width="1.5"/>
  <line x1="127" y1="47" x2="147" y2="47" stroke="#c62828" stroke-width="1.2"/>
  <line x1="127" y1="80" x2="147" y2="80" stroke="#c62828" stroke-width="1.2"/>
  <text x="137" y="220" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">Baseline</text>
  <text x="137" y="56"  text-anchor="middle" font-family="monospace" font-size="10" fill="#c62828">11.4</text>
  <!-- Governor bar: 4.7 -->
  <rect x="210" y="140" width="65" height="55" fill="#43a047"/>
  <!-- CI whiskers: 3.0–6.5 -->
  <line x1="242" y1="125" x2="242" y2="160" stroke="#1b5e20" stroke-width="1.5"/>
  <line x1="232" y1="125" x2="252" y2="125" stroke="#1b5e20" stroke-width="1.2"/>
  <line x1="232" y1="160" x2="252" y2="160" stroke="#1b5e20" stroke-width="1.2"/>
  <text x="242" y="220" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1b5e20" font-weight="700">Governor</text>
  <text x="242" y="134" text-anchor="middle" font-family="monospace" font-size="10" fill="#1b5e20" font-weight="700">4.7 ★</text>
  <!-- 59% label -->
  <text x="175" y="235" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#e65100" font-weight="700">≈ 59% reduction · CIs non-overlapping</text>

  <!-- RIGHT PANEL: Mode distribution pie-like bar -->
  <rect x="350" y="15" width="300" height="220" rx="10" fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>
  <text x="500" y="38" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#333">Governor Mode Distribution</text>
  <!-- Stacked horizontal bar for Governor -->
  <text x="370" y="68" font-family="sans-serif" font-size="11" fill="#333">Governor (STG active):</text>
  <!-- EXECUTE 30% -->
  <rect x="370" y="78" width="84" height="30" fill="#43a047" rx="3"/>
  <text x="412" y="98" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#fff" font-weight="700">EXECUTE 30%</text>
  <!-- VERIFY 64% -->
  <rect x="454" y="78" width="179" height="30" fill="#ff8f00" rx="3"/>
  <text x="543" y="98" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#fff" font-weight="700">VERIFY 64%</text>
  <!-- SAFE 6% -->
  <rect x="633" y="78" width="17" height="30" fill="#ef5350" rx="3"/>
  <text x="380" y="128" font-family="monospace" font-size="9" fill="#43a047">30% EXECUTE</text>
  <text x="380" y="142" font-family="monospace" font-size="9" fill="#ff8f00">64% VERIFY</text>
  <text x="380" y="156" font-family="monospace" font-size="9" fill="#ef5350"> 6% SAFE</text>
  <!-- Stacked bar for Baseline -->
  <text x="370" y="180" font-family="sans-serif" font-size="11" fill="#333">Baseline (no governor):</text>
  <rect x="370" y="188" width="280" height="30" fill="#ef9a9a" rx="3"/>
  <text x="510" y="208" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#b71c1c" font-weight="700">EXECUTE 100% (no gating)</text>
  <!-- caption -->
  <text x="340" y="255" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 8: MuJoCo results. Left: 59% violation reduction (non-overlapping CIs). Right: mode distribution.</text>
</svg>

Reproducible implementation: **https://github.com/dfeen87/stg-embodied-poc**

---

## 9 · Limitations

**Conservative supervision.** The governor enforces threshold-based switching to reduce instability but does not guarantee semantic correctness of LLM outputs. It mitigates divergence rather than proving task validity.

**Threshold sensitivity.** Performance depends on τ₁ and τ₂. While ablation studies reduce tuning bias, adaptive or theoretically derived thresholds remain future work.

**Simulator-to-hardware transfer.** The MuJoCo validation serves as a physics-grounded proxy. Real hardware deployment would require certified monitoring layers replacing the simulator oracle.

**LLM abstraction.** The embodied experiments use a deterministic mock LLM for reproducibility. Integration with external API-based LLMs may introduce latency, stochasticity, and distribution shift.

---

## 10 · Conclusion

We presented a **deterministic Spiral-Time Governor** that suppresses hallucination-driven divergence in LLM-assisted climbing and walking robotics via a transparent instability functional ΔΦ, deterministic threshold gating, and discrete Lyapunov boundedness under bounded disturbances.

The proposed governance layer is model-agnostic, does not modify LLM weights, and provides auditable execution control through explicit instability thresholds and mode switching (EXECUTE / VERIFY / SAFE).

A **two-layer evaluation protocol** was established, consisting of a controlled synthetic stochastic testbed and a complementary MuJoCo-based physics-grounded transfer validation without parameter retuning. The framework offers a deterministic safety supervision mechanism suitable for long-horizon deployment of LLM-assisted legged robotic systems.

<!-- FIGURE: Summary Visual -->
<svg width="100%" viewBox="0 0 680 130" xmlns="http://www.w3.org/2000/svg">
  <!-- Three pillars of the contribution -->
  <rect x="40"  y="20" width="175" height="88" rx="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1"/>
  <text x="127" y="46" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#0d47a1">Formal Guarantee</text>
  <text x="127" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">ISS-style Lyapunov</text>
  <text x="127" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">boundedness under</text>
  <text x="127" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1565c0">bounded disturbances</text>
  <rect x="252" y="20" width="175" height="88" rx="10" fill="#fff8e1" stroke="#f9a825" stroke-width="1"/>
  <text x="340" y="46" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#e65100">Deterministic Gating</text>
  <text x="340" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#f57f17">ΔΦ functional with</text>
  <text x="340" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#f57f17">auditable thresholds</text>
  <text x="340" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#f57f17">model-agnostic</text>
  <rect x="464" y="20" width="175" height="88" rx="10" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1"/>
  <text x="551" y="46" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#1b5e20">Empirical Validation</text>
  <text x="551" y="64" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e7d32">Synthetic testbed +</text>
  <text x="551" y="80" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e7d32">MuJoCo PoC</text>
  <text x="551" y="96" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e7d32">59% violation reduction</text>
  <text x="340" y="122" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Figure 9: The three pillars of the Spiral-Time Governor contribution.</text>
</svg>

---

## Acknowledgements

Repository scaffolding and documentation were developed with assistance from Claude (Sonnet 4.6) by Anthropic and Google (Jules Pro). All scientific content, mathematical formulations, and experimental design originate from the authors.

---

## Funding

This research received no external funding. The work was conducted independently by the authors.

---

## Data Availability

Full simulation source code, configuration files, seed lists, and statistical evaluation scripts are available at: **https://github.com/dfeen87/stg-embodied-poc**

---

## AI Usage Statement

LLMs were used solely as **research objects** within the experimental framework. Generative AI tools were additionally used for language refinement and minor formatting assistance. All mathematical derivations, experimental design, statistical analysis, and scientific conclusions were developed and verified by the authors. *The authors take full responsibility for the content of this manuscript.*

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

[1] D. Driess, F. Xia, M. T. Knoop, et al., "PaLM-E: An Embodied Multimodal Language Model," arXiv:2303.03378, 2023. doi:10.48550/arXiv.2303.03378

[2] P. Xu, "Embodied AI: Bridging Simulation and Reality in Robotics," Proceedings of the 4th International Symposium on Robotics, Artificial Intelligence and Information Engineering (RAIIE), 2025. doi:10.1109/RAIIE65740.2025.11140070

[3] X. Zhou et al., "Large Language Models for Robotics: Opportunities and Challenges," arXiv:2308.14455, 2023.

[4] S. Basu, M. H. Kim, S. Tatlidil, T. Williams, S. Sloman, and R. I. Bahar, "Augmenting large language models with psychologically grounded models of causal reasoning for planning under uncertainty," *Frontiers in Artificial Intelligence*, vol. 8, Art. 1730614, Jan. 2026. doi: 10.3389/frai.2025.1730614.

[5] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety-Critical Systems," *IEEE Transactions on Automatic Control*, vol. 62, no. 8, pp. 3861–3876, Aug. 2017. doi: 10.1109/TAC.2016.2638961.

[6] W. Huang et al., "Inner Monologue: Embodied Reasoning through Planning with Language Models," arXiv preprint arXiv:2207.05608, 2022.

[7] A. Z. Ren, B. Govil, T.-Y. Yang, K. Narasimhan, and A. Majumdar, "Robots That Ask for Help: Uncertainty-Aligned LLM Planning," arXiv preprint arXiv:2307.01928, 2023.

[8] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023. doi: 10.1145/3571730.

[9] L. Huang et al., "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," arXiv preprint arXiv:2311.05232, 2023.

[10] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language Agents with Iterative Design Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[11] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control Barrier Functions: Theory and Application," in *Proc. IEEE 18th European Control Conference (ECC)*, Naples, Italy, 2019, pp. 3420–3431. doi: 10.23919/ECC.2019.8796030.

[12] R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, "Safe Control with Learned Models: Optimality and Runtime Guarantees," *IEEE Transactions on Automatic Control*, 2023. doi: 10.1109/TAC.2023.3247173.

[13] S. Gu, J. Grudzien Kuba, Y. Chen, Y. Du, L. Yang, A. Knoll, and Y. Yang, "Safe multi-agent reinforcement learning for multi-robot control," *Artificial Intelligence*, vol. 319, 103905, 2023. doi:10.1016/j.artint.2023.103905.

[14] H. K. Khalil, *Nonlinear Systems*, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2002.

[15] J.-J. E. Slotine and W. Li, *Applied Nonlinear Control*. Englewood Cliffs, NJ, USA: Prentice Hall, 1991.

[16] E. D. Sontag, "Input-to-State Stability: Basic Concepts and Results," in *Nonlinear Dynamics and Operational Control*, P. Nistri and G. Stefani, Eds. Berlin, Germany: Springer, 1989, pp. 163–220. doi:10.1007/978-3-540-79026-86.

[17] Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, S. de Freitas, N. Heess, and M. Riedmiller, "dm_control: Software and Tasks for Continuous Control," arXiv:1801.00690, 2018. doi:10.48550/arXiv.1801.00690.

[18] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. doi:10.48550/arXiv.2005.11401.

[19] J. Liang, W. Huang, F. Xia, P. Florence, A. Zeng, "Code as Policies: Language Model Programs for Embodied Control," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2023. doi:10.1109/ICRA48891.2023.10160591.

[20] A. Brohan, N. Brown, B. Zitkovich et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv:2307.15818, 2023. doi:10.48550/arXiv.2307.15818
