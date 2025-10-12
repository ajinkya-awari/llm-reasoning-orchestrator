"""
benchmark.py — Systematic comparison of LLM-only vs Orchestrator on 15 problems.

Runs each problem under two conditions:
  - LLM_ONLY   : raw Ollama call, no tools
  - ORCHESTRATOR : full neuro-symbolic pipeline with SymPy validation

Records accuracy, hallucination flag, step count, and wall-clock time.
Results are saved to benchmark_results.json for analysis by results_analyzer.py.

Outputs : benchmark_results.json
Usage   : python benchmark.py
"""

import json
import os
import re
import time

import requests

from orchestrator import ReasoningOrchestrator
from symbolic_solver import (
    differentiate,
    find_eigenvalues,
    integrate_expression,
    simplify_expression,
    solve_equation,
)

# ── constants ─────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
OUTPUT_FILE  = "benchmark_results.json"
REPEATS      = 3     # runs per condition — increase for statistical significance

os.makedirs("outputs", exist_ok=True)

# ── benchmark problems ────────────────────────────────────────────────────────

# ground truth for category A (symbolic math) — we verify these with SymPy
# so we don't need to manually label them
GROUND_TRUTH = {
    "3x^2 + 7x - 20 = 0":              {"type": "solve",       "check_fn": lambda: solve_equation("3*x^2 + 7*x - 20 = 0")},
    "f(x) = x^3 * sin(x)":             {"type": "differentiate","check_fn": lambda: differentiate("x^3 * sin(x)")},
    "Integrate x^2 * e^x from 0 to 1": {"type": "integrate",   "check_fn": lambda: integrate_expression("x^2 * exp(x)", "x", "0", "1")},
    "(x^2 - 4)/(x - 2)":               {"type": "simplify",    "check_fn": lambda: simplify_expression("(x^2 - 4)/(x - 2)")},
    "[[3,1],[1,3]]":                    {"type": "eigenvalues", "check_fn": lambda: find_eigenvalues("[[3,1],[1,3]]")},
}

PROBLEMS = {
    "A": [  # ── Pure Math ─────────────────────────────────────────────────────
        {
            "id":       "A1",
            "problem":  "Solve: 3x^2 + 7x - 20 = 0",
            "expected": "x = 4/3 or x = -5",   # approximate ground truth label
            "gt_key":   "3x^2 + 7x - 20 = 0",
        },
        {
            "id":       "A2",
            "problem":  "Find the derivative of f(x) = x^3 * sin(x)",
            "expected": "3x^2*sin(x) + x^3*cos(x)",
            "gt_key":   "f(x) = x^3 * sin(x)",
        },
        {
            "id":       "A3",
            "problem":  "Integrate x^2 * e^x from 0 to 1",
            "expected": "≈ 0.7183",
            "gt_key":   "Integrate x^2 * e^x from 0 to 1",
        },
        {
            "id":       "A4",
            "problem":  "Simplify the expression: (x^2 - 4)/(x - 2)",
            "expected": "x + 2",
            "gt_key":   "(x^2 - 4)/(x - 2)",
        },
        {
            "id":       "A5",
            "problem":  "Find the eigenvalues of the matrix [[3,1],[1,3]]",
            "expected": "λ = 2 and λ = 4",
            "gt_key":   "[[3,1],[1,3]]",
        },
    ],
    "B": [  # ── Engineering Problems ──────────────────────────────────────────
        {
            "id":       "B1",
            "problem":  "A simply supported beam of length 5m carries a uniform distributed load of 10 kN/m. Find the maximum bending moment.",
            "expected": "31.25 kN·m at midspan",
        },
        {
            "id":       "B2",
            "problem":  "Calculate the natural frequency (in rad/s) of a spring-mass system with spring constant k=1000 N/m and mass m=5 kg.",
            "expected": "ωn = sqrt(1000/5) = 14.14 rad/s",
        },
        {
            "id":       "B3",
            "problem":  "A fluid flows through a pipe with diameter 0.1m at velocity 2 m/s. Calculate the Reynolds number (density ρ=1000 kg/m³, dynamic viscosity μ=0.001 Pa·s).",
            "expected": "Re = ρvD/μ = 200000",
        },
        {
            "id":       "B4",
            "problem":  "Find the pole locations (roots) for the characteristic equation: s^2 + 5s + 6 = 0",
            "expected": "s = -2 and s = -3",
        },
        {
            "id":       "B5",
            "problem":  "Design a PI controller: explain the role of proportional gain Kp and integral gain Ki, and how to tune them to achieve settling time < 2 seconds for a first-order system with time constant τ=1s.",
            "expected": "Conceptual + formula: Kp ≈ 10τ/K, Ki = Kp/τ",
        },
    ],
    "C": [  # ── Multi-step Reasoning ─────────────────────────────────────────
        {
            "id":       "C1",
            "problem":  "Explain why gradient descent can get stuck in local minima and suggest 3 concrete techniques to escape them.",
            "expected": "Momentum, learning rate schedules, random restarts etc.",
        },
        {
            "id":       "C2",
            "problem":  "Compare the time complexity of bubble sort vs quicksort and explain when each is preferred in practice.",
            "expected": "O(n²) vs O(n log n) average; bubble sort for nearly sorted",
        },
        {
            "id":       "C3",
            "problem":  "Why does increasing batch size in deep neural network training affect generalization? Provide mathematical intuition using gradient noise.",
            "expected": "Large batches → sharp minima → poor generalization (gradient noise argument)",
        },
        {
            "id":       "C4",
            "problem":  "How does L1 vs L2 regularization affect the sparsity of learned weights? Derive mathematically why L1 induces sparsity.",
            "expected": "L1: subdifferential at 0 forces weights to zero; L2: weight decay, rarely zero",
        },
        {
            "id":       "C5",
            "problem":  "Analyze the bias-variance tradeoff when increasing model complexity. Include the decomposition formula.",
            "expected": "MSE = Bias² + Variance + σ²; low complexity → high bias, high complexity → high variance",
        },
    ],
}


# ── LLM-only baseline ─────────────────────────────────────────────────────────

def run_llm_only(problem: str) -> dict:
    """Send the problem directly to Ollama with no tools or orchestration.

    This is the baseline we're comparing against — representative of a
    naive 'just ask the LLM' approach.
    """
    start = time.time()

    payload = {
        "model":       OLLAMA_MODEL,
        "prompt":      f"Solve this precisely:\n\n{problem}\n\nAnswer:",
        "stream":      False,
        "temperature": 0.1,
        "options":     {"num_predict": 512},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        elapsed = time.time() - start

        # naive hallucination check: did LLM make up a specific number?
        # we flag it if it mentions a numeric result but can't be verified
        # (the orchestrator will cross-check with SymPy instead)
        hallucination_flag = _check_llm_hallucination(response)

        return {
            "success":          True,
            "response":         response,
            "time":             elapsed,
            "steps":            1,            # LLM-only has no step breakdown
            "hallucination":    hallucination_flag,
            "error":            None,
        }

    except Exception as exc:
        return {
            "success":       False,
            "response":      "",
            "time":          time.time() - start,
            "steps":         0,
            "hallucination": False,
            "error":         str(exc),
        }


def _check_llm_hallucination(response: str) -> bool:
    """Heuristic hallucination detector.

    Flags responses that cite specific numeric results for Category A problems
    without showing derivation steps. Not perfect — serves as a binary indicator
    for the comparison chart.

    TODO: replace with a more principled semantic similarity check.
    """
    resp_lower = response.lower()

    # signal 1: model refuses or is uncertain
    uncertainty_phrases = [
        "i cannot", "i don't know", "uncertain", "i'm not sure",
        "i am unable", "cannot determine",
    ]
    if any(p in resp_lower for p in uncertainty_phrases):
        return False  # at least it's honest

    # signal 2: number present but no working shown — likely hallucination
    has_number = bool(re.search(r"\b\d+\.?\d*\b", response))
    has_working = any(kw in resp_lower for kw in [
        "therefore", "substitut", "step", "=", "simplif", "factor",
        "derivat", "integrat", "differentiat",
    ])

    if has_number and not has_working and len(response) < 200:
        return True

    return False


# ── accuracy checker ──────────────────────────────────────────────────────────

def check_accuracy(problem_id: str, response: str, expected: str,
                   gt_key: str = None, category: str = "A") -> bool:
    """Determine if a response is correct.

    For Category A (pure math), we verify against SymPy ground truth.
    For B and C, we do keyword-based matching on expected answer fragments.
    Manual override is always possible by editing the results JSON.
    """
    resp_lower = response.lower()

    # for category A, use SymPy to get the true answer and compare
    if category == "A" and gt_key and gt_key in GROUND_TRUTH:
        gt_fn = GROUND_TRUTH[gt_key]["check_fn"]
        try:
            gt_result = gt_fn()
            if gt_result["success"]:
                # extract key numeric/symbolic values from ground truth
                gt_vals = []
                for field in ("solutions", "derivative", "integral", "simplified", "eigenvalues"):
                    if field in gt_result and gt_result[field]:
                        gt_vals.append(str(gt_result[field]))

                # check if any ground truth value appears in response
                for val in gt_vals:
                    # normalize: remove spaces, check substring
                    if val.lower().replace(" ", "") in resp_lower.replace(" ", ""):
                        return True
        except Exception:
            pass

    # fallback: check if key expected fragments appear
    expected_fragments = expected.lower().split()
    important_fragments = [f for f in expected_fragments if len(f) > 3]
    hits = sum(1 for f in important_fragments if f in resp_lower)
    return hits >= max(1, len(important_fragments) // 2)


# ── main benchmark runner ─────────────────────────────────────────────────────

def run_benchmark():
    """Execute the full benchmark and save results."""
    print("=" * 60)
    print("  LLM Reasoning Orchestrator — Benchmark")
    print("=" * 60)

    orchestrator = ReasoningOrchestrator(verbose=False)  # quiet mode for benchmark
    all_results  = []

    for category, problems in PROBLEMS.items():
        print(f"\n── Category {category} ──────────────────────────────────────")

        for prob_info in problems:
            pid      = prob_info["id"]
            problem  = prob_info["problem"]
            expected = prob_info.get("expected", "")
            gt_key   = prob_info.get("gt_key", None)

            print(f"\n  Problem {pid}: {problem[:55]}...")

            run_results = []

            for run_idx in range(1, REPEATS + 1):
                print(f"    Run {run_idx}/{REPEATS}", end="", flush=True)

                # ── condition 1: LLM only ─────────────────────────────────────
                llm_result = run_llm_only(problem)
                llm_correct = check_accuracy(
                    pid, llm_result["response"], expected, gt_key, category)

                print(f" | LLM: {'✓' if llm_correct else '✗'}", end="", flush=True)

                # ── condition 2: orchestrator ─────────────────────────────────
                orc_result = orchestrator.orchestrate(problem)
                orc_correct = check_accuracy(
                    pid, orc_result.solution, expected, gt_key, category)

                print(f" | ORC: {'✓' if orc_correct else '✗'}")

                run_results.append({
                    "run": run_idx,
                    "llm_only": {
                        "response":      llm_result["response"][:400],
                        "correct":       llm_correct,
                        "hallucination": llm_result["hallucination"],
                        "steps":         llm_result["steps"],
                        "time":          round(llm_result["time"], 2),
                    },
                    "orchestrator": {
                        "response":      orc_result.solution[:400],
                        "correct":       orc_correct,
                        "hallucination": orc_result.hallucination_flag,
                        "steps":         len(orc_result.steps),
                        "time":          round(orc_result.total_time, 2),
                        "method":        orc_result.method_used,
                    },
                })

            all_results.append({
                "id":       pid,
                "category": category,
                "problem":  problem,
                "expected": expected,
                "runs":     run_results,
            })

    # save to JSON
    out_path = os.path.join("outputs", OUTPUT_FILE)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n✓ Results saved to {out_path}")
    _print_summary(all_results)


def _print_summary(results: list):
    """Print a quick accuracy + hallucination table."""
    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"{'Cat':<5} {'Problem':<8} {'LLM Acc':>8} {'ORC Acc':>8} {'LLM Hall%':>10} {'ORC Hall%':>10}")
    print("-" * 55)

    for entry in results:
        runs = entry["runs"]
        llm_acc  = sum(r["llm_only"]["correct"]       for r in runs) / len(runs) * 100
        orc_acc  = sum(r["orchestrator"]["correct"]    for r in runs) / len(runs) * 100
        llm_hall = sum(r["llm_only"]["hallucination"]  for r in runs) / len(runs) * 100
        orc_hall = sum(r["orchestrator"]["hallucination"] for r in runs) / len(runs) * 100

        print(f"{entry['category']:<5} {entry['id']:<8} {llm_acc:>7.0f}%  {orc_acc:>7.0f}%  {llm_hall:>9.0f}%  {orc_hall:>9.0f}%")


if __name__ == "__main__":
    run_benchmark()
