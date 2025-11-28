"""
results_analyzer.py — Visualise benchmark results from benchmark.py.

Reads benchmark_results.json and produces three publication-quality figures:
  1. accuracy_by_category.png  — grouped bar chart, LLM-only vs Orchestrator
  2. hallucination_comparison.png — hallucination rate per condition
  3. reasoning_pipeline.png    — flowchart of the orchestration architecture

Outputs : outputs/accuracy_by_category.png
          outputs/hallucination_comparison.png
          outputs/reasoning_pipeline.png
Usage   : python results_analyzer.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

RESULTS_FILE = os.path.join("outputs", "benchmark_results.json")
OUT_DIR      = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# paper-quality styling — mimicking IEEE/NeurIPS plots
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":    150,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "llm":  "#E07B54",   # warm orange — LLM only
    "orc":  "#3B82B8",   # steel blue — orchestrator
}

CATEGORY_NAMES = {
    "A": "Pure Math\n(n=5)",
    "B": "Engineering\n(n=5)",
    "C": "Reasoning\n(n=5)",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_results() -> list:
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run benchmark.py first or use generate_synthetic_results() below.")
        return []
    with open(RESULTS_FILE) as f:
        return json.load(f)


def generate_synthetic_results() -> list:
    """Generate plausible synthetic results for README / pre-run visualisation.

    Based on expected performance differences from the paper (Verma 2025) and
    the ControlBench study (Kevian et al. 2024). Replace with real data after
    running benchmark.py.
    """
    # [llm_accuracy, orc_accuracy, llm_hallucination, orc_hallucination]
    synthetic = {
        "A": {"llm_acc": 48, "orc_acc": 91, "llm_hall": 42, "orc_hall": 4},
        "B": {"llm_acc": 55, "orc_acc": 78, "llm_hall": 35, "orc_hall": 8},
        "C": {"llm_acc": 72, "orc_acc": 74, "llm_hall": 18, "orc_hall": 12},
    }
    return synthetic


def aggregate_results(results: list) -> dict:
    """Compute per-category accuracy and hallucination rate."""
    agg = {}
    for entry in results:
        cat = entry["category"]
        if cat not in agg:
            agg[cat] = {"llm_correct": [], "orc_correct": [],
                        "llm_hall":    [], "orc_hall":    []}

        for run in entry["runs"]:
            agg[cat]["llm_correct"].append(int(run["llm_only"]["correct"]))
            agg[cat]["orc_correct"].append(int(run["orchestrator"]["correct"]))
            agg[cat]["llm_hall"].append(int(run["llm_only"]["hallucination"]))
            agg[cat]["orc_hall"].append(int(run["orchestrator"]["hallucination"]))

    summary = {}
    for cat, vals in agg.items():
        summary[cat] = {
            "llm_acc":  np.mean(vals["llm_correct"]) * 100,
            "orc_acc":  np.mean(vals["orc_correct"]) * 100,
            "llm_hall": np.mean(vals["llm_hall"]) * 100,
            "orc_hall": np.mean(vals["orc_hall"]) * 100,
        }
    return summary


# ── plot 1: accuracy by category ─────────────────────────────────────────────

def plot_accuracy(summary: dict, synthetic: bool = False):
    """Grouped bar chart: LLM-only vs Orchestrator accuracy per category."""
    cats  = sorted(summary.keys())
    x     = np.arange(len(cats))
    width = 0.35

    llm_vals = [summary[c]["llm_acc"] for c in cats]
    orc_vals = [summary[c]["orc_acc"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, llm_vals, width, label="LLM-Only (Baseline)",
                   color=COLORS["llm"], alpha=0.88, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, orc_vals, width, label="AI Reasoning Orchestrator",
                   color=COLORS["orc"], alpha=0.88, edgecolor="white", linewidth=0.8)

    # annotate bars with percentage
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Problem Category")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison: LLM-Only vs AI Reasoning Orchestrator")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_NAMES.get(c, c) for c in cats])
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if synthetic:
        ax.text(0.98, 0.02, "* Synthetic data — run benchmark.py for real results",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


# ── plot 2: hallucination rate ────────────────────────────────────────────────

def plot_hallucination(summary: dict, synthetic: bool = False):
    """Side-by-side bar chart of hallucination rates."""
    cats     = sorted(summary.keys())
    x        = np.arange(len(cats))
    width    = 0.35

    llm_hall = [summary[c]["llm_hall"] for c in cats]
    orc_hall = [summary[c]["orc_hall"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width/2, llm_hall, width, label="LLM-Only",
           color=COLORS["llm"], alpha=0.88, edgecolor="white")
    ax.bar(x + width/2, orc_hall, width, label="Orchestrator",
           color=COLORS["orc"], alpha=0.88, edgecolor="white")

    for i, (lv, ov) in enumerate(zip(llm_hall, orc_hall)):
        reduction = lv - ov
        if reduction > 0:
            ax.annotate(f"−{reduction:.0f}%",
                        xy=(x[i] + width/2, ov + 1),
                        ha="center", va="bottom", fontsize=9,
                        color="#2d6a2d", fontweight="bold")

    ax.set_xlabel("Problem Category")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Rate: LLM-Only vs AI Reasoning Orchestrator")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_NAMES.get(c, c) for c in cats])
    ax.set_ylim(0, 60)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if synthetic:
        ax.text(0.98, 0.02, "* Synthetic data — run benchmark.py for real results",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hallucination_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


# ── plot 3: pipeline flowchart ────────────────────────────────────────────────

def plot_pipeline():
    """Visualise the orchestration pipeline as a clean flowchart.

    Done with matplotlib patches — no graphviz dependency required.
    This is essentially Figure 1 from Verma (2025) re-implemented.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(x, y, w, h, label, color, fontsize=10, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                wrap=True, zorder=4)

    def arrow(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.1, label, ha="left", va="bottom",
                    fontsize=8, color=color)

    # nodes
    box(0.3, 3.0, 2.2, 1.0, "User Problem\n(NL Input)", "#607D8B")
    box(3.0, 3.0, 2.2, 1.0, "Decompose\n(LLM)", "#5C6BC0")
    box(5.8, 3.0, 2.2, 1.0, "Classify\nStep Type", "#7B1FA2", text_color="white")
    box(3.5, 1.0, 2.2, 1.0, "SymPy Solver\n(Exact Math)", "#2E7D32")
    box(7.5, 1.0, 2.2, 1.0, "LLM Reasoning\n(Conceptual)", "#1565C0")
    box(5.8, 5.0, 2.2, 1.0, "Validate\nStep", "#6D4C41")
    box(9.5, 3.0, 2.2, 1.0, "Synthesize\nFinal Answer", "#C62828")

    # arrows
    arrow(2.5, 3.5, 3.0, 3.5)
    arrow(5.2, 3.5, 5.8, 3.5)
    arrow(6.9, 3.0, 5.0, 2.0, "symbolic")    # classify → sympy
    arrow(6.9, 3.0, 8.0, 2.0, "reasoning")   # classify → llm
    arrow(5.0, 2.0, 5.8, 5.0, "result")       # sympy → validate
    arrow(8.0, 2.0, 7.0, 5.0, "result")       # llm → validate
    arrow(7.0, 5.5, 9.5, 3.8, "all steps\nvalidated")

    ax.set_title("AI Reasoning Orchestrator — Architecture Pipeline",
                 fontsize=13, fontweight="bold", pad=15, color="#1a1a2e")

    # legend
    legend_items = [
        mpatches.Patch(color="#2E7D32", label="SymPy (exact)"),
        mpatches.Patch(color="#1565C0", label="LLM (semantic)"),
        mpatches.Patch(color="#7B1FA2", label="Classifier"),
        mpatches.Patch(color="#6D4C41", label="Validator"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "reasoning_pipeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Results Analyzer ──\n")

    raw = load_results()
    using_synthetic = False

    if not raw:
        print("Using synthetic data for visualisation...")
        summary = generate_synthetic_results()
        using_synthetic = True
    else:
        summary = aggregate_results(raw)
        print(f"Loaded {len(raw)} benchmark entries from {RESULTS_FILE}")

    plot_accuracy(summary,      synthetic=using_synthetic)
    plot_hallucination(summary, synthetic=using_synthetic)
    plot_pipeline()

    print("\n✓ All plots saved to outputs/")
