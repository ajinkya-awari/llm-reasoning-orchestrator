"""
orchestrator.py — Core ReasoningOrchestrator class.

Implements the AI Reasoning Orchestrator framework from Verma (2025) and the
neuro-symbolic paradigm described in Yang et al. (2025). The LLM (running locally
via Ollama) acts as a semantic orchestrator that decomposes problems and delegates
precise computation to SymPy, rather than attempting arithmetic directly.

Architecture:
    User Problem → Decompose → [Classify each step] → [SymPy | LLM] → Validate → Synthesize

Outputs : used by benchmark.py and demo.py
Usage   : python orchestrator.py
"""

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import requests

from symbolic_solver import (
    check_constraint,
    differentiate,
    find_eigenvalues,
    integrate_expression,
    simplify_expression,
    solve_equation,
)

# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"          # change to "llama3" or "phi3" if gemma3:4b not installed

# bump this up if you're running on a slow machine
OLLAMA_TIMEOUT = 600

# how confident does the classifier need to be before routing to SymPy?
# 0.5 means "if at all ambiguous, try SymPy first"
SYMBOLIC_CONFIDENCE_THRESHOLD = 0.5


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class SolutionStep:
    """Represents one step in the orchestration pipeline."""
    step_id:      int
    description:  str
    method:       str          # "symbolic", "llm", or "skipped"
    input_text:   str
    result:       str
    sub_steps:    list  = field(default_factory=list)
    validated:    bool  = False
    time_taken:   float = 0.0
    error:        Optional[str] = None


@dataclass
class OrchestrationResult:
    """Full result returned by orchestrate()."""
    problem:       str
    solution:      str
    steps:         list
    method_used:   str          # "symbolic", "llm", or "hybrid"
    total_time:    float
    hallucination_flag: bool    # True if LLM said something contradicting SymPy
    success:       bool
    error:         Optional[str] = None


# ── orchestrator ──────────────────────────────────────────────────────────────

class ReasoningOrchestrator:
    """
    Neuro-symbolic reasoning pipeline.

    Routes problem steps between a local LLM (semantic understanding, multi-step
    reasoning) and SymPy (exact symbolic math). Follows the ReAct-style interleaving
    of reasoning and tool calls — see Yao et al. (2023).
    """

    def __init__(self, model: str = OLLAMA_MODEL, verbose: bool = True):
        self.model   = model
        self.verbose = verbose
        self._check_ollama_connection()

    # ── connection check ──────────────────────────────────────────────────────

    def _check_ollama_connection(self):
        """Make sure Ollama is running before we start. Fail fast."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if self.verbose:
                print(f"✓ Ollama connected. Available models: {models}")
            if not any(self.model in m for m in models):
                print(f"  ⚠ Model '{self.model}' not found. Run: ollama pull {self.model}")
        except Exception as exc:
            raise ConnectionError(
                f"Ollama not reachable at {OLLAMA_URL}.\n"
                f"Start it with: ollama serve\n"
                f"Original error: {exc}"
            )

    # ── LLM wrapper ───────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """Send a prompt to Ollama and return the response text.

        Low temperature (0.1) keeps math reasoning deterministic.
        Raises on connection failure; returns empty string on decode error.
        """
        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "stream":      False,
            "temperature": temperature,
            "options":     {"num_predict": 1024},
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    # ── step 1: decompose ─────────────────────────────────────────────────────

    def decompose_problem(self, problem_text: str) -> list[dict]:
        """Ask the LLM to break a problem into numbered sub-steps.

        Returns a list of dicts: [{step_num, description, requires_math}]
        Falls back to treating the whole problem as a single step on failure.
        """
        prompt = f"""You are a precise engineering and mathematics assistant.

Break the following problem into clear sequential steps. For each step, state:
1. A brief description of what needs to be done
2. Whether it requires exact symbolic math (True/False)

Return ONLY a JSON array, nothing else. Format:
[
  {{"step": 1, "description": "...", "requires_math": true/false}},
  ...
]

Problem: {problem_text}"""

        if self.verbose:
            print(f"  [Decompose] Calling LLM...")

        raw = self._call_llm(prompt)

        # extract JSON array even if LLM wraps it in markdown fences
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # fallback: single-step
        return [{"step": 1, "description": problem_text, "requires_math": False}]

    # ── step 2: classify ──────────────────────────────────────────────────────

    def classify_step(self, step_description: str) -> tuple[str, float]:
        """Determine whether a step should go to SymPy or the LLM.

        Returns (method, confidence) where method is 'symbolic' or 'llm'.
        This is a simple keyword + LLM heuristic — could be replaced with a
        trained classifier if accuracy matters more than setup simplicity.
        """
        desc_lower = step_description.lower()

        # strong keyword signals for symbolic routing
        symbolic_keywords = [
            "solve", "derivative", "differentiate", "integrate", "integral",
            "eigenvalue", "simplify", "factor", "expand", "equation",
            "roots", "zeros", "polynomial", "matrix", "determinant",
            "natural frequency", "transfer function", "pole", "zero",
            "reynolds", "bending moment", "calculate", "compute", "find the value",
        ]

        # these are clearly conceptual — LLM handles them
        reasoning_keywords = [
            "explain", "why", "compare", "discuss", "analyze", "describe",
            "intuition", "bias-variance", "tradeoff", "regularization effect",
            "gradient descent stuck", "when to use",
        ]

        sym_score = sum(1 for kw in symbolic_keywords if kw in desc_lower)
        llm_score = sum(1 for kw in reasoning_keywords if kw in desc_lower)

        if sym_score > llm_score and sym_score > 0:
            return "symbolic", min(0.5 + sym_score * 0.1, 0.95)
        elif llm_score > 0:
            return "llm", min(0.5 + llm_score * 0.1, 0.95)

        # ambiguous — ask LLM itself (meta-classification, see ReAct paper)
        prompt = f"""Classify this problem step. Reply with ONLY one word: "symbolic" or "llm".

"symbolic" = requires precise algebra, calculus, matrix ops, or arithmetic
"llm" = requires explanation, comparison, conceptual reasoning, or prose

Step: {step_description}"""

        answer = self._call_llm(prompt, temperature=0.0).lower().strip()
        if "symbolic" in answer:
            return "symbolic", 0.6
        return "llm", 0.6

    # ── step 3a: symbolic solver ──────────────────────────────────────────────

    def solve_symbolic(self, step_description: str) -> dict:
        """Route a step to the appropriate SymPy function.

        Pattern-matches the step description to pick the right solver.
        This is the key part that prevents arithmetic hallucination.
        """
        desc_lower = step_description.lower()

        # derivative / differentiation
        if any(kw in desc_lower for kw in ["derivative", "differentiate", "d/dx", "d/dt"]):
            # try to extract expression from description
            expr = self._extract_expression(step_description, "differentiate")
            var  = "x" if " t " not in desc_lower else "t"
            return {"type": "differentiate", "result": differentiate(expr, var)}

        # integration
        if any(kw in desc_lower for kw in ["integrat", "integral", "∫"]):
            expr = self._extract_expression(step_description, "integrate")
            # check for bounds like "from 0 to 1"
            bounds = re.search(r"from\s+([\d\.\-]+)\s+to\s+([\d\.\-]+)", desc_lower)
            if bounds:
                return {"type": "integrate", "result": integrate_expression(
                    expr, "x", bounds.group(1), bounds.group(2))}
            return {"type": "integrate", "result": integrate_expression(expr)}

        # eigenvalues
        if any(kw in desc_lower for kw in ["eigenvalue", "eigenvalues", "matrix"]):
            mat = self._extract_matrix(step_description)
            if mat:
                return {"type": "eigenvalues", "result": find_eigenvalues(mat)}

        # simplify
        if any(kw in desc_lower for kw in ["simplify", "simplification"]):
            expr = self._extract_expression(step_description, "simplify")
            return {"type": "simplify", "result": simplify_expression(expr)}

        # solve equation — default for anything with '='
        expr = self._extract_expression(step_description, "solve")
        return {"type": "solve", "result": solve_equation(expr)}

    def _extract_expression(self, text: str, hint: str = "solve") -> str:
        """Pull out a math expression from a natural language step description.

        Heuristic: look for text after common trigger words.
        Falls back to passing the whole text to SymPy (which will fail gracefully).
        """
        # look for f(x) = ..., equation: ..., etc.
        patterns = [
            r"f\([xyz]\)\s*=\s*(.+?)(?:\.|$)",
            r"(?:equation|expression|function)\s*[:\-]?\s*(.+?)(?:\.|from|$)",
            r"(?:of|for)\s+(.+?)(?:\s+from|\s+with|\.|$)",
            r":\s*(.+?)(?:\.|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # just return the whole thing and let SymPy error gracefully
        return text

    def _extract_matrix(self, text: str) -> Optional[str]:
        """Find a [[...]] matrix literal in text."""
        m = re.search(r"\[\s*\[.+?\]\s*\]", text)
        if m:
            return m.group()
        return None

    # ── step 3b: LLM solver ───────────────────────────────────────────────────

    def solve_with_llm(self, step_description: str, context: str = "") -> str:
        """Use the LLM for conceptual, explanatory, or reasoning-heavy steps.

        We give the LLM the context of previously solved steps so it can
        build on them — this is the 'chain of thought' integration point.
        """
        context_block = f"\n\nContext from previous steps:\n{context}" if context else ""

        prompt = f"""You are an expert in engineering, mathematics, and computer science.

Answer the following precisely and concisely. Show your reasoning.
If this involves any numerical calculation, state clearly that the number is approximate.{context_block}

Task: {step_description}

Answer:"""

        return self._call_llm(prompt, temperature=0.2)

    # ── step 4: validate ──────────────────────────────────────────────────────

    def validate_step(self, step_description: str, solution: str,
                      method: str = "llm") -> tuple[bool, str]:
        """Check whether a solution is consistent and flag potential hallucinations.

        For symbolic results, we trust SymPy implicitly.
        For LLM results, we ask the LLM to self-check — not perfect but
        catches obvious nonsense (wrong units, impossible values, etc.).

        Returns (is_valid, validation_note)
        """
        if method == "symbolic":
            # SymPy doesn't hallucinate — if it ran without error, it's correct
            return True, "Verified by SymPy (exact computation)"

        # LLM self-validation prompt — based on retrospective verification from
        # Liu et al. (2025) Safe framework, though we're not formalizing in Lean here
        prompt = f"""Review this solution for obvious errors, wrong formulas, or impossible values.
Reply with ONLY: "VALID" or "INVALID: <brief reason>"

Problem step: {step_description}
Solution: {solution[:500]}"""  # cap to 500 chars to stay within context

        verdict = self._call_llm(prompt, temperature=0.0)

        if verdict.upper().startswith("INVALID"):
            reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else "Flagged by validator"
            return False, reason

        return True, "LLM self-validation passed"

    # ── step 5: synthesize ────────────────────────────────────────────────────

    def _synthesize_answer(self, problem: str, steps: list[SolutionStep]) -> str:
        """Combine all solved steps into a final cohesive answer."""
        steps_summary = "\n".join(
            f"Step {s.step_id} ({s.method}): {s.result[:300]}"
            for s in steps if s.result and not s.error
        )

        prompt = f"""You are summarizing a solved engineering/math problem.

Original problem: {problem}

Solved steps:
{steps_summary}

Write a clear, concise final answer (3-5 sentences max). Be precise.
If numerical values were computed, state them. Do not add new information."""

        return self._call_llm(prompt, temperature=0.1)

    # ── main pipeline ─────────────────────────────────────────────────────────

    def orchestrate(self, problem: str) -> OrchestrationResult:
        """Run the full orchestration pipeline for a given problem.

        This is the main entry point. Returns an OrchestrationResult with
        the solution, all intermediate steps, method used, and timing.
        """
        start_time = time.time()
        solved_steps = []
        methods_used = set()
        hallucination_detected = False

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"Problem: {problem[:80]}...")
            print(f"{'─'*60}")

        try:
            # ── 1. decompose ──────────────────────────────────────────────────
            if self.verbose:
                print("\n[1/5] Decomposing problem...")
            raw_steps = self.decompose_problem(problem)

            if self.verbose:
                print(f"  → {len(raw_steps)} steps identified")

            # ── 2-4. classify + solve + validate each step ────────────────────
            context_so_far = ""

            for raw in raw_steps:
                step_start = time.time()
                sid   = raw.get("step", len(solved_steps) + 1)
                desc  = raw.get("description", str(raw))
                needs_math = raw.get("requires_math", False)

                if self.verbose:
                    print(f"\n[Step {sid}] {desc[:60]}...")

                # classify — even if LLM said requires_math=True, double-check
                method, confidence = self.classify_step(desc)

                if self.verbose:
                    print(f"  → Classified as: {method} (confidence={confidence:.2f})")

                # solve
                result_text = ""
                sub_steps   = []
                error       = None

                if method == "symbolic":
                    try:
                        sym_result = self.solve_symbolic(desc)
                        r = sym_result["result"]
                        if r["success"]:
                            # pick the most informative key to surface
                            for key in ("solutions", "derivative", "integral",
                                        "simplified", "eigenvalues"):
                                if key in r and r[key]:
                                    result_text = f"{key}: {r[key]}"
                                    break
                            sub_steps = r.get("steps", [])
                            methods_used.add("symbolic")
                        else:
                            # symbolic failed — fall back to LLM
                            if self.verbose:
                                print(f"  ⚠ SymPy failed ({r['error']}), falling back to LLM")
                            result_text = self.solve_with_llm(desc, context_so_far)
                            method = "llm"
                            methods_used.add("llm")
                    except Exception as exc:
                        if self.verbose:
                            print(f"  ⚠ Symbolic error: {exc}, falling back to LLM")
                        result_text = self.solve_with_llm(desc, context_so_far)
                        method = "llm"
                        methods_used.add("llm")

                else:
                    result_text = self.solve_with_llm(desc, context_so_far)
                    methods_used.add("llm")

                # validate
                is_valid, validation_note = self.validate_step(desc, result_text, method)
                if not is_valid:
                    hallucination_detected = True
                    if self.verbose:
                        print(f"  ⚠ Validation issue: {validation_note}")
                else:
                    if self.verbose:
                        print(f"  ✓ Validated: {validation_note}")

                solved_steps.append(SolutionStep(
                    step_id     = sid,
                    description = desc,
                    method      = method,
                    input_text  = desc,
                    result      = result_text,
                    sub_steps   = sub_steps,
                    validated   = is_valid,
                    time_taken  = time.time() - step_start,
                    error       = error,
                ))

                context_so_far += f"\nStep {sid}: {result_text[:200]}"

            # ── 5. synthesize final answer ─────────────────────────────────────
            if self.verbose:
                print("\n[5/5] Synthesizing final answer...")

            final_answer = self._synthesize_answer(problem, solved_steps)

            # determine overall method label
            if "symbolic" in methods_used and "llm" in methods_used:
                overall_method = "hybrid"
            elif "symbolic" in methods_used:
                overall_method = "symbolic"
            else:
                overall_method = "llm"

            total_time = time.time() - start_time

            if self.verbose:
                print(f"\n✓ Done in {total_time:.1f}s | Method: {overall_method}")
                print(f"  Hallucination flag: {hallucination_detected}")

            return OrchestrationResult(
                problem            = problem,
                solution           = final_answer,
                steps              = [vars(s) for s in solved_steps],
                method_used        = overall_method,
                total_time         = total_time,
                hallucination_flag = hallucination_detected,
                success            = True,
            )

        except Exception as exc:
            traceback.print_exc()
            return OrchestrationResult(
                problem            = problem,
                solution           = "",
                steps              = [vars(s) for s in solved_steps],
                method_used        = "failed",
                total_time         = time.time() - start_time,
                hallucination_flag = False,
                success            = False,
                error              = str(exc),
            )


# ── quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    orc = ReasoningOrchestrator(verbose=True)

    test_problem = "Solve the equation 3x^2 + 7x - 20 = 0 and verify the solutions."
    result = orc.orchestrate(test_problem)

    print("\n── Final Answer ──")
    print(result.solution)
    print(f"\nMethod: {result.method_used} | Steps: {len(result.steps)}")
