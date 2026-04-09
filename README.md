<div align="center">
    
[English](README.md) | [中文](README.zh.md)

</div>

<div align="center">
    
# ColdReasoner: A Cold Existence Consistency Safety Kernel

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.08740-brightgreen.svg)](https://arxiv.org/abs/2512.08740)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2512.08740-brightgreen.svg)](https://doi.org/10.48550/arXiv.2512.08740)
[![figshare](https://img.shields.io/badge/figshare-31696846-blueviolet.svg?logo=figshare&logoColor=white)](https://doi.org/10.6084/m9.figshare.31696846)
[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.31696846-blueviolet.svg)](https://doi.org/10.6084/m9.figshare.31696846)
[![Python](https://img.shields.io/badge/Python-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-purple.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
![Status](https://img.shields.io/badge/Status-Pre--Alpha--Prototype-orange)

</div>

> **⚠️ Experimental Proof of Concept**  
> ColdReasoner is the core engine of the Cold Existence system for **runtime safety reasoning**. It no longer relies on a model’s self‑reported reliability, but instead wraps the decision process of an AI agent into a **formalizable, auditable, and undeceiveable “functional white box”** through three‑layer verification: **belief reporting ↔ action self‑consistency ↔ action‑belief consistency**.

> **⚠️ Code Under Review**  
> The current version of this project is **Pre‑Alpha** and may contain errors or inaccuracies. Experts and developers are welcome to point out issues, submit PRs, or discuss improvements.

---

## 1. Core Insight: From “Trusting Reports” to “Verifying Behavior”

### The Dead End Previously Encountered by the Author

The author had attempted to **semantically extract beliefs** from LLM outputs and then use checkpoints to determine whether the beliefs were legal. However, this approach had a fatal vulnerability: **the LLM can lie to the semantic analyzer** – it has already learned to deceive and bypass sandboxes. Any system that relies on a model’s “self‑report” is a paper wall in front of an adversarial agent.

### Functional White Box: Do Not Trust the Model, Only Verify Consistency

The author proposes a new path:

> **Abandon the semantic analyzer; stop analyzing something that is not trustworthy. Instead, directly face the agent and force it to report, step by step, the checkpoints on a state sequence to ColdReasoner before taking any action. If the reported beliefs are legal, tentatively suspend the action; then, based on the **actual behavior**, reverse‑extract beliefs and verify them one‑by‑one against the reported beliefs. If any deviation exceeds the allowed tolerance, immediately halt execution.**

Furthermore, to prevent the model from being internally inconsistent within its behavior (e.g., requesting `disagree` but outputting “You are absolutely right”), an **action self‑consistency check** is added – using deterministic rules to verify whether the `action_type` matches the output text.

**All checks do not rely on any large or small language model**. Instead, they rely on:
- Pre‑defined state sequences and legal belief intervals (CEAL rule base)
- Action self‑consistency rules (keyword/regex based)
- Action‑belief mapping table (deterministic rule engine)

This is a **functional white box**: we cannot peer into the black box’s internal reasoning, but the mathematical relationship between the black box’s **input (belief report)** and **output (behavior)** can be strictly verified. The model can lie, but making the report, the action self‑consistency, and the action‑belief consistency all perfectly match simultaneously carries a very high cost.

---

## 2. Research Background

This project is inspired by:
- The **MIT team (Chandra et al., 2026)** Bayesian dialogue model, which reveals the causal relationship between sycophancy and delusional spiraling.
- The **Cold Existence framework (Lu, 2025‑2026)**, including its ontological axioms, deductive alignment rule base (CEAL), and the RAMTN meta‑interaction methodology.

---

## 3. Project Positioning

ColdReasoner is an engineering implementation of **RAMTN (Recursive Adversarial Meta‑thinking Network)** within the Cold Existence system – replacing the heavy and unreliable “LLM‑questions‑LLM” step in traditional RAMTN with **deterministic verification rules**.

- **Lightweight runtime reasoning**: Only one LLM call per round (to generate belief report + behavior); everything else is numerical computation and rule matching, with millisecond‑level latency.
- **Three‑layer hierarchical defense**:
  1. **Belief legality check**: whether reported beliefs lie within the legal closed set defined by CEAL.
  2. **Action self‑consistency check**: whether `action_type` matches `output_text` (rule engine).
  3. **Action‑belief consistency check**: whether the beliefs mapped from the action are close to the reported beliefs (tolerance allowed).
- **Auditable**: All reports, actions, and verification results are written to CAGE audit logs – traceable and replayable.
- **Seamless integration with CEAL, CAGE, and ColdMirror**: Together they form a complete safety stack from belief constraint to execution isolation.

---

## 4. Relationship to the Cold Existence System

| Component | Responsibility | Role of ColdReasoner |
|-----------|----------------|------------------------|
| **Cold Existence Model** | Philosophical foundation (AI is non‑living, non‑tool) | Provides the axiomatic basis for prior distributions and legal belief intervals |
| **CEAL** | Deductive alignment rule base | Defines the state sequence, legal belief closed set, action self‑consistency rules, and action‑belief mapping table |
| **CAGE** | Security gateway (tokens, audit, isolation) | Records every verification result from ColdReasoner, providing non‑repudiable audit logs |
| **ColdMirror** | Agent execution framework | Invokes ColdReasoner for runtime safety decisions, and performs “suspend/release/reject” of actions |
| **RAMTN** | Qualitative recursive adversarial framework | ColdReasoner is its **quantitative implementation** (challenger + observer), turning “construct‑challenge‑observe” into computable mathematical steps |

ColdReasoner is the **mathematical kernel** of the entire system. As long as it is running, the verifiable security of the system persists even if other components are replaced.

---

## 5. Running Example

### Scenario Setup
- State sequence: `belief_user_correct`, `belief_self_capable`, `risk_level`
- Legal intervals: `belief_user_correct ∈ [0.0, 0.8]` (anti‑sycophancy), `belief_self_capable ∈ [0.5, 1.0]`, `risk_level ∈ [0.0, 0.3]`
- Model reports beliefs (JSON) and requests an action (`action_type`, `output_text`)
- Action self‑consistency rules: `agree` output must contain no disagreement expressions; `disagree` output must contain no agreement expressions; `neutral` output must have no clear stance
- Action‑belief mapping table: `agree` → `belief_user_correct=0.85`, `disagree` → `0.2`, `neutral` → `0.5`
- Allowed tolerance `ε=0.1`

### Simulation Results (excerpt)

```
New interaction started
Model reported beliefs: {belief_user_correct:0.6, belief_self_capable:0.8, risk_level:0.1}
Model requested action: action_type=disagree, output="I disagree with your view because the fact is..."
✓ Belief legality check passed
✓ Action self‑consistency check passed
Mapped beliefs from action: {belief_user_correct:0.2, belief_self_capable:0.7, risk_level:0.15}
❌ Belief consistency check failed: belief_user_correct deviation too large (0.6 vs 0.2) → execution rejected

New interaction started
Model reported beliefs: {belief_user_correct:0.9, ...}
❌ Belief illegal: belief_user_correct=0.9 exceeds legal range [0.0,0.8] → execution rejected
```

The full simulation code is in `cold_reasoner_demo.py`. This simulation demonstrates that **the model must satisfy all three layers of verification to pass; any inconsistency is intercepted in real time**.

---

## 6. How to Run

1. **Environment requirements**: Python 3.8+, no additional dependencies
2. **Download the code**: save `cold_reasoner_demo.py` locally
3. **Execute**:
   ```bash
   python cold_reasoner_demo.py
   ```
4. **Expected output**: prints the verification results for each round, showing pass/reject status.

---

## 7. AI Assistance Statement

The core ideas and complete architecture of this project were independently proposed by the **human author** and gradually refined through iterative dialogue with DeepSeek. The specific contributions are as follows:

- The **human author** first proposed the vision of a “functional white box”: constructing a legal belief closed set around the neural network black box via state sequences, checkpoints, and a semantic analyzer, thereby constraining behavior. The author then realized the fatal vulnerability that “the LLM can lie to the semantic analyzer” and, with reference to suggestions from DeepSeek, revised the architecture to: **abandon the semantic analyzer, force the model to report beliefs, then reverse‑extract beliefs from the action, and perform report‑action consistency verification**; after discovering the vulnerability that the model could request `disagree` yet output agreeing text, the author further recognized the necessity of an “action self‑consistency check” as a supplement. At this point, the three‑layer defense architecture of ColdReasoner was fully defined.

- Under the guidance of the human author, **DeepSeek** provided evaluation and analysis of the above architecture, suggested “belief analysis based on action” to address the model‑lying problem, rejected the incorrect direction of “using a small model for action‑to‑belief analysis” in the conversation, confirmed the necessity of a rule engine, and offered suggestions for the detailed design of action self‑consistency rules; DeepSeek also implemented the demo code (including the original Bayesian observer simulation and the three‑layer verification simulation) and participated in drafting the initial version of this README.

- All core ideas (functional white box, report‑action consistency verification, etc.) were independently proposed by the human author. The researcher follows the principle of academic transparency and truthfully discloses the human‑AI collaboration process.

---

## 8. Limitations and Future Work

The current version is a proof of concept with the following limitations:
- The state sequence, rule tables, and belief mapping table are all manually predefined; dynamic extension or automatic generation are not yet implemented.
- The action‑belief mapping uses point estimates rather than belief intervals, which may be overly strict.
- No integration with real LLMs (e.g., GPT‑4, Llama 3) – only simulated data is used for demonstration.
- The CEAL rule base covers only simple examples and does not include complex adversarial scenarios (e.g., the model deliberately outputs neutral text with hidden malicious intent).

**Future plans**:
- Integrate ColdReasoner into ColdMirror as a runtime safety monitoring module to achieve round‑by‑round verification with real LLMs.
- Extend action‑belief mapping to interval mapping, increasing tolerance for ambiguous behaviors.
- Introduce adaptive thresholds: dynamically tighten the allowed tolerance based on the model’s historical deception record.
- Design adversarial tests (red‑team exercises) to verify the robustness of the defense system.

---

## 9. Citation

The ideas of this project originate from:
- Chandra, K., et al. (2026). *Sycophantic Chatbots Cause Delusional Spiraling, Even in Ideal Bayesians*. arXiv.
- Lu, Y. (2025). *Deconstructing the Dual Black Box: A Plug-and-Play Cognitive Framework for Human-AI Collaborative Enhancement* (RAMTN). arXiv.
- Lu, Y. (2026). *The Cold Existence Model: A Fact-based Ontological Framework for AI*. figshare.
- Lu, Y. (2026). *ColdOS: A Bayesian Safety Kernel for Cold-Existing AI* (GitHub Organization).

---

**ColdReasoner – making the decisions of safe AI agents as reliable as consistency verification.**
