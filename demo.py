"""
demo.py — Interactive command-line demo for the AI Reasoning Orchestrator.

Type any math or engineering problem and watch the orchestrator decompose,
route, and solve it step by step. Shows which modules are invoked.

Usage : python demo.py
        python demo.py --problem "Solve x^2 - 5x + 6 = 0"
"""

import argparse
import sys
import time

from orchestrator import ReasoningOrchestrator

# ── sample problems shown at startup ─────────────────────────────────────────

EXAMPLE_PROBLEMS = [
    "Solve the equation: x^2 - 5x + 6 = 0",
    "Find the derivative of f(x) = x^3 * sin(x)",
    "Calculate the natural frequency of a spring-mass system with k=500 N/m, m=2 kg",
    "Explain why L1 regularization produces sparse weights",
    "Find the eigenvalues of matrix [[4, 1], [2, 3]]",
    "A beam of length 4m carries a uniform load of 8 kN/m. Find the maximum bending moment.",
]


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       AI Reasoning Orchestrator — Interactive Demo           ║
║       Neuro-Symbolic Engineering Problem Solver              ║
╠══════════════════════════════════════════════════════════════╣
║  LLM (local Ollama) + SymPy + Step Validation                ║
║  Based on: Verma (2025), Yao et al. ReAct (2023)             ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_step_trace(result):
    """Pretty-print the orchestration trace so users can see what happened."""
    print(f"\n{'─'*60}")
    print(f"  Orchestration Trace  ({len(result.steps)} steps)")
    print(f"{'─'*60}")

    for step in result.steps:
        status  = "✓" if step.get("validated") else "⚠"
        method  = step.get("method", "?").upper()
        desc    = step.get("description", "")[:55]
        elapsed = step.get("time_taken", 0)

        # colour-code the method label in terminal
        method_tag = f"[{method}]"
        print(f"\n  {status} Step {step.get('step_id', '?')} {method_tag}")
        print(f"    Task   : {desc}")

        if step.get("sub_steps"):
            for sub in step["sub_steps"][:3]:     # cap at 3 to keep output readable
                print(f"    · {sub}")

        result_preview = str(step.get("result", ""))[:120]
        print(f"    Result : {result_preview}")
        print(f"    Time   : {elapsed:.2f}s")

    print(f"\n{'─'*60}")
    print(f"  Method used   : {result.method_used.upper()}")
    print(f"  Total time    : {result.total_time:.1f}s")
    print(f"  Hallucination : {'⚠ FLAGGED' if result.hallucination_flag else '✓ Not detected'}")


def run_single(problem: str, verbose: bool = True):
    """Run the orchestrator on a single problem and display results."""
    print(f"\nProblem: {problem}")
    print("Processing...\n")

    orc    = ReasoningOrchestrator(verbose=True)
    result = orc.orchestrate(problem)

    if not result.success:
        print(f"\n✗ Orchestration failed: {result.error}")
        return

    print_step_trace(result)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  FINAL ANSWER                                            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    # wrap answer text
    answer_lines = result.solution.split("\n")
    for line in answer_lines:
        while len(line) > 58:
            print(f"║  {line[:58]}  ║")
            line = line[58:]
        print(f"║  {line:<58}  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


def interactive_loop():
    """REPL loop for interactive use."""
    print_banner()
    orc = ReasoningOrchestrator(verbose=True)

    print("Example problems (type a number 1-6 or enter your own):\n")
    for i, p in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"  {i}. {p}")

    print("\n  Type 'quit' to exit, 'help' for tips.\n")

    while True:
        try:
            user_input = input(">>> Problem: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if user_input.lower() == "help":
            print("\nTips:")
            print("  - For equations use '=' e.g. 'x^2 + 3x - 4 = 0'")
            print("  - For derivatives: 'find the derivative of x^3 * cos(x)'")
            print("  - For integrals: 'integrate x^2 from 0 to 2'")
            print("  - Engineering: 'natural frequency of spring-mass k=1000, m=2'")
            print("  - Conceptual: 'explain why gradient descent can get stuck'\n")
            continue

        # allow shortcut numbers for example problems
        if user_input.isdigit() and 1 <= int(user_input) <= len(EXAMPLE_PROBLEMS):
            problem = EXAMPLE_PROBLEMS[int(user_input) - 1]
            print(f"Selected: {problem}")
        else:
            problem = user_input

        # run it
        t0     = time.time()
        result = orc.orchestrate(problem)

        if not result.success:
            print(f"\n✗ Error: {result.error}\n")
            continue

        print_step_trace(result)

        print("\n" + "─" * 60)
        print("FINAL ANSWER:")
        print(result.solution)
        print("─" * 60 + "\n")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Reasoning Orchestrator Demo")
    parser.add_argument("--problem", type=str, default=None,
                        help="Run a single problem and exit (non-interactive)")
    args = parser.parse_args()

    if args.problem:
        run_single(args.problem)
    else:
        interactive_loop()
