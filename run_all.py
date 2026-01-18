"""
run_all.py — Master runner for the AI Reasoning Orchestrator project.

Executes the full pipeline in the correct order:
  1. symbolic_solver.py self-test
  2. orchestrator.py smoke test
  3. benchmark.py (15 problems, ~10-15 min depending on hardware)
  4. results_analyzer.py (generate all plots)

Run this after setting up the environment to reproduce all results.

Usage : python run_all.py
        python run_all.py --skip-benchmark   (if you already have results)
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("Symbolic Solver Self-Test",   "symbolic_solver.py",  False),
    ("Orchestrator Smoke Test",      "orchestrator.py",     False),
    ("Benchmark (15 problems)",      "benchmark.py",        True),   # can skip
    ("Results & Visualisation",      "results_analyzer.py", False),
]


def run_step(name: str, script: str) -> bool:
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed:.0f}s")
        return True
    else:
        print(f"\n✗ {name} FAILED (exit code {result.returncode})")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip benchmark.py (use existing results)")
    args = parser.parse_args()

    print("\nAI Reasoning Orchestrator — Full Pipeline Run")
    print(f"Python: {sys.version.split()[0]}")

    all_ok = True
    for name, script, can_skip in STEPS:
        if can_skip and args.skip_benchmark:
            print(f"\n  Skipping: {name}")
            continue
        ok = run_step(name, script)
        if not ok:
            all_ok = False
            print(f"\n  Stopping due to failure in: {name}")
            break

    print("\n" + "─"*60)
    if all_ok:
        print("✓ All steps completed. Check outputs/ for plots and results.")
    else:
        print("✗ Pipeline did not complete. Check errors above.")
