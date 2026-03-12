# LLM Reasoning Orchestrator
### A Neuro-Symbolic Architecture for Structured Engineering Reasoning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![SymPy](https://img.shields.io/badge/SymPy-1.13-green)
![HuggingFace](https://img.shields.io/badge/LLM-TinyLlama%201.1B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Motivation

Large language models demonstrate strong performance on natural language reasoning tasks, yet fail in a consistent and predictable pattern: they produce confident, fluent answers to problems requiring exact arithmetic or symbolic manipulation that are simply wrong. This failure mode arithmetic hallucination is not random noise. It reflects a structural mismatch between how autoregressive language models process information and what exact computation actually requires.

Kevian et al. (ControlBench, 2024) documented this systematically in control engineering contexts, showing that LLMs fail not because they lack domain knowledge but because they cannot reliably execute the computational steps that domain knowledge calls for. The model may correctly identify that a problem requires the quadratic formula, then evaluate it incorrectly.

This raises a question that is architectural rather than scaling-based: is the solution to use a larger model, or to redesign how computation is organized? Recent work on tool-augmented LLMs particularly ReAct (Yao et al., ICLR 2023) and Toolformer (Schick et al., NeurIPS 2023) suggests that routing specific sub-tasks to external tools can recover reliability without increasing model size. This project investigates that hypothesis in the context of structured engineering and mathematics problems.

---

## Research Questions

**RQ1 Routing effectiveness:** Does routing computation-heavy steps to a symbolic engine (SymPy) improve accuracy on exact mathematics tasks, compared to asking the LLM to solve the same steps directly?

**RQ2 Category specificity:** Is the accuracy improvement from symbolic routing concentrated on pure mathematical sub-tasks, or does it generalize to applied engineering and conceptual reasoning problems?

**RQ3 Hallucination and routing:** Does step-level validation in the orchestration pipeline reduce the rate of confident incorrect outputs, and does this effect vary across problem types?

**RQ4 Decomposition quality:** When a small language model (TinyLlama-1.1B) is used as the orchestrator, how reliably does it decompose multi-step problems into sub-steps that can be correctly classified and routed?

---

## Methodology

### Experimental Design

Each problem was evaluated under two conditions:

- **LLM-Only:** the base model receives the full problem and generates a direct answer
- **Orchestrator:** the same model decomposes the problem, routes each step to either SymPy or the LLM, validates each result, and synthesizes a final answer

Both conditions used the same underlying model (TinyLlama-1.1B-Chat) to isolate the effect of architecture rather than model capacity.

### Problem Set

15 problems across three categories, each run 3 times per condition (45 LLM calls per condition total):

| Category | Description | n |
|---|---|---|
| A  Pure Math | Equations, derivatives, integrals, simplification, eigenvalues | 5 |
| B  Engineering | Beam bending, natural frequency, Reynolds number, transfer function poles, PI control | 5 |
| C  Reasoning | Gradient descent, sorting complexity, batch size effects, L1/L2 regularization, bias-variance | 5 |

Category A problems have exact verifiable ground truth (checked against SymPy). Categories B and C are evaluated by keyword matching against expected concepts in the correct answer.

### Orchestration Pipeline

```
Problem P
    │
    ▼
[1] Decompose into steps s₁...sₙ   (LLM)
    │
    ▼
[2] Classify each step              (keyword routing)
    │
    ├──► symbolic  ──► SymPy exact solver
    │                        │
    └──► reasoning ──► LLM chain-of-thought
                             │
    ◄────────────────────────┘
[3] Validate step result           (SymPy check or LLM self-validation)
    │
    ▼
[4] Synthesize final answer        (LLM with full step context)
```

### Models and Infrastructure

- **Language model:** TinyLlama-1.1B-Chat-v1.0 (HuggingFace, open weights)
- **Symbolic engine:** SymPy 1.13 with implicit multiplication parser
- **Hardware:** Google Colab Tesla T4 GPU
- **Cost:** $0 entirely free infrastructure

### Evaluation Metric

Binary accuracy per run: 1 if the response contains the expected answer concepts, 0 otherwise. Reported as mean accuracy across 3 runs per problem. Hallucination flagged when step-level validation fails.

---

## Key Findings

**Finding 1 Symbolic routing provides the largest gains precisely where LLMs are weakest.**
On pure math problems (Category A), the LLM-only baseline scored 0% accuracy not due to lack of knowledge about the methods, but due to failure in execution. The orchestrated pipeline raised this to 40% by delegating computation to SymPy. This is not a marginal improvement; it is the difference between complete failure and partial success on the task type where arithmetic hallucination is most expected.

**Finding 2 The effect is category-specific, not universal.**
On conceptual reasoning problems (Category C), both conditions scored 60% identical performance. Symbolic tools provide no advantage where exact computation is not required. The fact that the gap is zero here and large on Category A supports the hypothesis that routing improves reliability through structural means, not general capability enhancement.

**Finding 3 Engineering problems show intermediate improvement.**
Category B improved from 20% to 40%. These problems mix symbolic computation (e.g. solving a characteristic equation for pole locations) with domain knowledge application. The partial improvement suggests the pipeline successfully routes computational sub-components while remaining bottlenecked by the LLM's engineering domain knowledge.

**Finding 4 Decomposition quality limits the pipeline at small model scale.**
TinyLlama-1.1B frequently decomposed multi-step problems into a single step, reducing the routing classifier's opportunity to separate symbolic from reasoning sub-tasks. The orchestration benefit is bounded by decomposition granularity, which is itself a function of the orchestrating model's instruction-following capability.

### Results Summary

| Category | LLM-Only | Orchestrator | Δ |
|---|---|---|---|
| A  Pure Math (n=5) | 0% | **40%** | +40 pp |
| B  Engineering (n=5) | 20% | **40%** | +20 pp |
| C  Reasoning (n=5) | 60% | 60% | 0 pp |
| **Overall** | **27%** | **47%** | **+20 pp** |

---

## Visual Results

Three figures are included in `outputs/`:

**`accuracy_by_category.png`**  grouped bar chart comparing LLM-only vs Orchestrator accuracy across the three problem categories. The zero-to-40% gap on Category A is the central visual result.

**`hallucination_comparison.png`**  hallucination rate by category and condition. Both conditions show 0% hallucination with TinyLlama, consistent with the model's tendency toward conservative outputs under uncertainty.

**`reasoning_pipeline.png`**  architecture diagram illustrating the decompose → classify → route → validate → synthesize flow with color-coded pathways for symbolic and LLM routing.

---

## Limitations

**Model scale.** TinyLlama-1.1B was chosen for zero-cost reproducibility. A more capable orchestrator (e.g. Mistral-7B or Llama-3-8B) would likely produce finer-grained decompositions and better routing decisions, potentially widening the accuracy gap on Categories A and B.

**Routing classifier.** The current classifier uses keyword matching with a fixed vocabulary. It is brittle to paraphrasing "find the roots" routes correctly while "determine where the function crosses zero" may not. A learned classifier would generalize more robustly.

**Evaluation breadth.** 15 problems is sufficient to observe directional trends but not to establish statistical significance. A larger benchmark with held-out test problems would strengthen the conclusions.

**Accuracy metric.** Binary keyword-match accuracy is coarse for Categories B and C. Rubric-based evaluation or human annotation would provide a more reliable signal on reasoning quality.

---

## Future Work

**Decomposition quality as an independent variable.** Run the same pipeline with orchestrators of increasing capability (TinyLlama-1.1B → Mistral-7B → Llama-3-8B) while holding the symbolic engine constant. This would isolate the contribution of decomposition quality to the overall accuracy gain and clarify whether the current results underestimate the architecture's potential.

**Learned routing classifier.** Replace keyword matching with a small trained classifier that predicts symbolic vs reasoning routing from step descriptions. This connects directly to work on tool-use learning in language models and would address the brittleness identified above.

**Formal verification integration.** Extend the validation step from SymPy checking to formal proof verification using Z3 or a fragment of Lean 4. This would allow stronger correctness guarantees on engineering constraint problems and aligns with research on verified AI reasoning.

**Benchmark expansion.** Extend to control systems problems (transfer functions, stability analysis, Bode plots) and convex optimization problems (KKT conditions, duality) categories where the gap between LLM knowledge and LLM computation is particularly pronounced, as documented in ControlBench.

**Communication-constrained orchestration.** Investigate how the pipeline degrades when LLM inference is subject to latency or bandwidth constraints a natural extension toward distributed inference settings where orchestration calls carry non-trivial communication cost.

---

## Repository Structure

```
llm-reasoning-orchestrator/
├── orchestrator.py         # ReasoningOrchestrator  full pipeline
├── symbolic_solver.py      # SymPy wrapper (solve, diff, integrate, eigenvalues)
├── benchmark.py            # 15-problem benchmark, 3 runs per condition
├── results_analyzer.py     # Matplotlib figures
├── demo.py                 # Interactive CLI with step-level trace
├── run_all.py              # Master runner
├── requirements.txt
└── outputs/
    ├── benchmark_results.json
    ├── accuracy_by_category.png
    ├── hallucination_comparison.png
    └── reasoning_pipeline.png
```

---

## Reproducing the Results

```bash
git clone https://github.com/ajinkya-awari/llm-reasoning-orchestrator
cd llm-reasoning-orchestrator
pip install -r requirements.txt
```

Run a single problem:
```bash
python demo.py --problem "Solve x^2 - 5x + 6 = 0"
```

Run the full benchmark (~25 minutes on Colab T4 GPU):
```bash
python benchmark.py
python results_analyzer.py
```

The benchmark requires a GPU or significant patience on CPU. Google Colab free tier (T4 runtime) is the recommended environment.

---

## References

1. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
2. Schick et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS 2023.
3. Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
4. Yang et al. *Neuro-Symbolic AI: A Systematic Review.* arXiv:2508.13678, 2025.
5. Kevian et al. *Capabilities of Large Language Models in Control Engineering.* arXiv:2404.03647, 2024.
6. Chu et al. *A Survey of Chain of Thought Reasoning.* arXiv:2309.15402, 2024.

---

*Part of a research portfolio for MSc applications in Machine Learning and Computer Science.*  
*Related: [Neural Network Optimizer Study](https://github.com/ajinkya-awari) · [GNN Agricultural Networks](https://github.com/ajinkya-awari)*
