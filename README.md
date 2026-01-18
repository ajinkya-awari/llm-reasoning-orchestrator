# AI Reasoning Orchestrator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![SymPy](https://img.shields.io/badge/SymPy-1.13-green)
![HuggingFace](https://img.shields.io/badge/LLM-TinyLlama%201.1B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Abstract

Large Language Models exhibit impressive natural language understanding yet fail
systematically on tasks requiring exact arithmetic and symbolic manipulation — a
well-documented phenomenon known as arithmetic hallucination (Kevian et al., 2024).
This project implements the **AI Reasoning Orchestrator** framework, a neuro-symbolic
architecture where the LLM acts as a semantic orchestrator that decomposes complex
engineering and mathematics problems, routes computation-heavy sub-tasks to a SymPy
symbolic engine, and validates each step before synthesis.

Evaluation across 15 problems (5 pure math, 5 engineering, 5 multi-step reasoning)
shows the orchestrated pipeline achieves **40% accuracy on pure math vs 0% for
LLM-only**, confirming that symbolic routing provides the largest gains precisely
where LLMs are weakest — exact computation.

---

## Problem Formulation

Let a complex problem P be decomposable into n sub-steps. Each step is classified
by a routing function R: S → {symbolic, llm}.

For symbolic steps, SymPy computes an exact answer:

    a_i = SymPy(s_i)          if R(s_i) = symbolic

For reasoning steps, the LLM generates a response with chain-of-thought context:

    a_i = LLM(s_i | a_1...a_{i-1})    if R(s_i) = llm

A validation function V(s_i, a_i) checks each result. This extends the ReAct
framework (Yao et al., 2023) with an explicit symbolic computation pathway.

---

## Architecture
```
User Problem → Decompose → Classify → Route → Validate → Final Answer
                (LLM)      (Keywords)   |
                                        ├── SymPy (exact math)
                                        └── LLM  (reasoning)
```

---

## Experimental Results

**Model:** TinyLlama-1.1B-Chat | **Platform:** Google Colab T4 GPU | **Runs:** 3 per problem

### Accuracy (%) by Category

| Category              | LLM-Only | Orchestrator | Improvement |
|-----------------------|----------|--------------|-------------|
| A — Pure Math (n=5)   | 0%       | **40%**      | +40 pp      |
| B — Engineering (n=5) | 20%      | **40%**      | +20 pp      |
| C — Reasoning (n=5)   | 60%      | 60%          | 0 pp        |
| **Overall**           | **27%**  | **47%**      | **+20 pp**  |

### Key Finding

The orchestrator's advantage is largest on pure math (Category A), where the
LLM-only baseline scores 0% due to arithmetic brittleness. The symbolic routing
to SymPy raises this to 40% — consistent with the core hypothesis that
neuro-symbolic routing prevents hallucination on computation-heavy tasks.

On conceptual reasoning (Category C), both systems perform identically (60%),
confirming that symbolic tools add value only when computation is required.

### Hallucination Rate

Both conditions show 0% hallucination with TinyLlama — the model is conservative
and avoids fabricating specific numerical results it is uncertain about.

---

## SymPy Capabilities

| Function | Example | Result |
|---|---|---|
| solve_equation | 3x²+7x-20=0 | x = 5/3 or x = -4 |
| differentiate | x³·sin(x) | x²(x·cos(x)+3·sin(x)) |
| integrate_expression | x²·eˣ from 0→1 | ≈ 0.7183 |
| simplify_expression | (x²-4)/(x-2) | x+2 |
| find_eigenvalues | [[3,1],[1,3]] | λ=2, λ=4 |

---

## Repository Structure
```
llm-reasoning-orchestrator/
├── orchestrator.py         # ReasoningOrchestrator — core pipeline
├── symbolic_solver.py      # SymPy wrapper
├── benchmark.py            # 15-problem benchmark
├── results_analyzer.py     # Plot generation
├── demo.py                 # Interactive CLI demo
├── run_all.py              # Master runner
├── requirements.txt
├── LICENSE
└── outputs/
    ├── benchmark_results.json
    ├── accuracy_by_category.png
    ├── hallucination_comparison.png
    └── reasoning_pipeline.png
```

---

## Setup
```bash
git clone https://github.com/ajinkya-awari/llm-reasoning-orchestrator
cd llm-reasoning-orchestrator
pip install -r requirements.txt
python demo.py --problem "Solve x^2 - 5x + 6 = 0"
```

---

## Connection to Research

| Framework | Paper | Relevance |
|---|---|---|
| LLM as orchestrator | Yao et al., ReAct, ICLR 2023 | Core architecture |
| Tool-augmented LLMs | Schick et al., Toolformer, NeurIPS 2023 | Tool invocation |
| Chain-of-Thought | Wei et al., CoT, NeurIPS 2022 | Decomposition |
| Neuro-Symbolic AI | Yang et al., arXiv 2025 | NeSy formulation |
| LLM hallucination | Kevian et al., ControlBench, arXiv 2024 | Motivation |

This hybrid neuro-symbolic approach aligns with TUM's research on formal methods
in AI and the goal of building systems with verifiable correctness guarantees.

---

## References

1. Yao et al. ReAct: Synergizing Reasoning and Acting. ICLR 2023.
2. Schick et al. Toolformer. NeurIPS 2023.
3. Wei et al. Chain-of-Thought Prompting. NeurIPS 2022.
4. Yang et al. Neuro-Symbolic AI. arXiv:2508.13678, 2025.
5. Kevian et al. ControlBench. arXiv:2404.03647, 2024.

---

## Citation
```bibtex
@software{awari2025orchestrator,
  author = {Awari, Ajinkya},
  title  = {AI Reasoning Orchestrator: Neuro-Symbolic Engineering Problem Solver},
  year   = {2025},
  url    = {https://github.com/ajinkya-awari/llm-reasoning-orchestrator}
}
```

## License

MIT
