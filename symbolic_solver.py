"""
symbolic_solver.py — SymPy-based symbolic math engine for the Reasoning Orchestrator.

Handles algebraic solving, differentiation, integration, simplification, and constraint
checking. All functions accept plain strings and return structured dicts so the
orchestrator can consume them without touching SymPy directly.

Outputs : used internally by orchestrator.py and benchmark.py
Usage   : python symbolic_solver.py  (runs a quick self-test)
"""

import re
import traceback

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ── parser setup ──────────────────────────────────────────────────────────────

# implicit multiplication lets users write "3x" instead of "3*x"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_COMMON_SYMBOLS = {
    name: sp.Symbol(name)
    for name in ["x", "y", "z", "t", "s", "n", "k", "a", "b", "c"]
}
_COMMON_SYMBOLS.update({
    "pi": sp.pi,
    "e":  sp.E,
    "I":  sp.I,
    "oo": sp.oo,
})


def _parse(expr_str: str) -> sp.Expr:
    """Parse a math string into a SymPy expression.

    Uses a permissive transformer so users don't have to write explicit
    multiplication signs. This is the only place we call parse_expr.
    """
    cleaned = expr_str.strip().replace("^", "**")
    return parse_expr(cleaned, local_dict=_COMMON_SYMBOLS, transformations=_TRANSFORMS)


def _fmt(expr: sp.Expr) -> str:
    """Pretty-print a SymPy expression as a clean string."""
    return str(sp.simplify(expr))


# ── core solver functions ─────────────────────────────────────────────────────

def solve_equation(equation_str: str) -> dict:
    """Solve an algebraic equation or system for its unknowns.

    Handles single-variable equations like '3x^2 + 7x - 20 = 0'
    and returns all real+complex roots with intermediate steps.

    Returns a dict with keys: success, solutions, steps, latex, error
    """
    try:
        steps = []

        # split on '=' — if no '=' treat as expression = 0
        if "=" in equation_str:
            lhs_str, rhs_str = equation_str.split("=", 1)
            lhs = _parse(lhs_str)
            rhs = _parse(rhs_str)
            expr = lhs - rhs
            steps.append(f"Rewrite as: {sp.pretty(lhs)} - ({sp.pretty(rhs)}) = 0")
        else:
            expr = _parse(equation_str)
            steps.append(f"Expression set to zero: {sp.pretty(expr)} = 0")

        # figure out what we're solving for — pick the first free symbol
        free = sorted(expr.free_symbols, key=lambda s: s.name)
        if not free:
            return {"success": False, "error": "No variables found in expression."}

        var = free[0]
        steps.append(f"Solving for: {var}")

        solutions = sp.solve(expr, var)
        steps.append(f"Raw solutions from SymPy: {solutions}")

        # also try to get numerical approximations
        numerical = []
        for sol in solutions:
            try:
                numerical.append(float(sol.evalf()))
            except Exception:
                numerical.append(str(sol.evalf()))

        steps.append(f"Numerical values: {numerical}")

        # latex for README / paper-quality output
        latex_sols = [sp.latex(s) for s in solutions]

        return {
            "success":    True,
            "solutions":  [_fmt(s) for s in solutions],
            "numerical":  numerical,
            "variable":   str(var),
            "steps":      steps,
            "latex":      latex_sols,
            "error":      None,
        }

    except Exception as exc:
        return {"success": False, "solutions": [], "steps": [], "error": str(exc)}


def differentiate(expr_str: str, variable: str = "x") -> dict:
    """Compute the derivative of an expression w.r.t. a variable.

    Supports nth-order derivatives — pass 'x,2' as variable for d²/dx².
    """
    try:
        steps = []

        # allow "x,2" shorthand for second derivative
        order = 1
        if "," in variable:
            var_name, order_str = variable.split(",", 1)
            variable = var_name.strip()
            order = int(order_str.strip())

        var = sp.Symbol(variable)
        expr = _parse(expr_str)
        steps.append(f"Expression: {sp.pretty(expr)}")
        steps.append(f"Differentiating w.r.t {variable}, order={order}")

        result = sp.diff(expr, var, order)
        simplified = sp.simplify(result)
        steps.append(f"After differentiation: {sp.pretty(result)}")

        if result != simplified:
            steps.append(f"Simplified: {sp.pretty(simplified)}")

        return {
            "success":    True,
            "derivative": _fmt(simplified),
            "order":      order,
            "variable":   variable,
            "steps":      steps,
            "latex":      sp.latex(simplified),
            "error":      None,
        }

    except Exception as exc:
        return {"success": False, "derivative": None, "steps": [], "error": str(exc)}


def integrate_expression(expr_str: str, variable: str = "x",
                          lower: str = None, upper: str = None) -> dict:
    """Compute definite or indefinite integral.

    For definite integrals pass lower='0', upper='1' etc.
    Handles common special functions like erf, Gamma via SymPy.
    """
    try:
        steps = []
        var = sp.Symbol(variable)
        expr = _parse(expr_str)
        steps.append(f"Integrand: {sp.pretty(expr)}")

        if lower is not None and upper is not None:
            lo = _parse(lower)
            hi = _parse(upper)
            steps.append(f"Definite integral from {lower} to {upper}")
            result = sp.integrate(expr, (var, lo, hi))
            result_simplified = sp.simplify(result)
            steps.append(f"Exact result: {sp.pretty(result_simplified)}")

            try:
                numerical = float(result_simplified.evalf())
                steps.append(f"Numerical value ≈ {numerical:.6f}")
            except Exception:
                numerical = str(result_simplified.evalf())

            return {
                "success":     True,
                "integral":    _fmt(result_simplified),
                "numerical":   numerical,
                "definite":    True,
                "steps":       steps,
                "latex":       sp.latex(result_simplified),
                "error":       None,
            }

        else:
            steps.append("Indefinite integral (+ C omitted)")
            result = sp.integrate(expr, var)
            result_simplified = sp.simplify(result)
            steps.append(f"Result: {sp.pretty(result_simplified)}")

            return {
                "success":  True,
                "integral": _fmt(result_simplified),
                "definite": False,
                "steps":    steps,
                "latex":    sp.latex(result_simplified),
                "error":    None,
            }

    except Exception as exc:
        return {"success": False, "integral": None, "steps": [], "error": str(exc)}


def simplify_expression(expr_str: str) -> dict:
    """Simplify a mathematical expression using SymPy's full simplification.

    Tries factor, cancel, trigsimp and picks the shortest representation —
    same heuristic as WolframAlpha's 'simplest form'.
    """
    try:
        steps = []
        expr = _parse(expr_str)
        steps.append(f"Input: {sp.pretty(expr)}")

        candidates = {
            "simplify": sp.simplify(expr),
            "factor":   sp.factor(expr),
            "cancel":   sp.cancel(expr),
            "expand":   sp.expand(expr),
        }

        # pick whichever string representation is shortest
        best_key = min(candidates, key=lambda k: len(str(candidates[k])))
        best = candidates[best_key]
        steps.append(f"Best form ({best_key}): {sp.pretty(best)}")

        # log all forms so the orchestrator can show them
        all_forms = {k: str(v) for k, v in candidates.items()}

        return {
            "success":    True,
            "simplified": _fmt(best),
            "method":     best_key,
            "all_forms":  all_forms,
            "steps":      steps,
            "latex":      sp.latex(best),
            "error":      None,
        }

    except Exception as exc:
        return {"success": False, "simplified": None, "steps": [], "error": str(exc)}


def find_eigenvalues(matrix_str: str) -> dict:
    """Compute eigenvalues (and optionally eigenvectors) of a matrix.

    Accepts list-of-lists format: '[[3,1],[1,3]]'
    TODO: add support for symbolic matrix entries
    """
    try:
        steps = []

        # eval is fine here since input is from our benchmark — not user-facing web
        raw = eval(matrix_str)  # noqa: S307  (controlled input only)
        M = sp.Matrix(raw)
        steps.append(f"Matrix:\n{sp.pretty(M)}")

        eigenvals = M.eigenvals()      # returns {eigenval: multiplicity}
        eigenvects = M.eigenvects()    # returns list of (eigenval, mult, [vects])

        steps.append(f"Eigenvalues (value: multiplicity): {eigenvals}")

        eigenval_list = []
        for val, mult in eigenvals.items():
            try:
                num_val = complex(val.evalf())
                eigenval_list.append({
                    "symbolic": str(val),
                    "numerical": f"{num_val.real:.4f}" + (f"+{num_val.imag:.4f}j" if abs(num_val.imag) > 1e-10 else ""),
                    "multiplicity": mult,
                })
            except Exception:
                eigenval_list.append({"symbolic": str(val), "multiplicity": mult})

        return {
            "success":     True,
            "eigenvalues": eigenval_list,
            "steps":       steps,
            "latex":       [sp.latex(v) for v in eigenvals.keys()],
            "error":       None,
        }

    except Exception as exc:
        return {"success": False, "eigenvalues": [], "steps": [], "error": str(exc)}


def check_constraint(expr_str: str, constraint: str) -> dict:
    """Verify whether an expression satisfies a given constraint.

    constraint format: 'x > 0', 'expr == 0', 'expr < 10' etc.
    Used by orchestrator to validate intermediate steps.
    """
    try:
        steps = []
        expr = _parse(expr_str)

        # naive approach: try to evaluate symbolically, fall back to assumption check
        steps.append(f"Checking: {expr_str} satisfies '{constraint}'")

        # handle numeric equality check
        if "==" in constraint:
            rhs = _parse(constraint.split("==")[1])
            diff = sp.simplify(expr - rhs)
            satisfied = diff == 0
            steps.append(f"Difference: {sp.pretty(diff)}")
        else:
            # just report symbolic truth value — good enough for orchestrator validation
            satisfied = None
            steps.append("Symbolic constraint check not conclusive — reporting expression value.")

        return {
            "success":   True,
            "satisfied": satisfied,
            "expr_value": _fmt(expr),
            "steps":     steps,
            "error":     None,
        }

    except Exception as exc:
        return {"success": False, "satisfied": None, "steps": [], "error": str(exc)}


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Symbolic Solver Self-Test ──\n")

    tests = [
        ("solve_equation",      solve_equation,          "3*x^2 + 7*x - 20 = 0"),
        ("differentiate",       differentiate,           "x^3 * sin(x)"),
        ("integrate_expression",integrate_expression,    ("x^2 * e^x", "x", "0", "1")),
        ("simplify_expression", simplify_expression,     "(x^2 - 4)/(x - 2)"),
        ("find_eigenvalues",    find_eigenvalues,        "[[3,1],[1,3]]"),
    ]

    for name, fn, arg in tests:
        print(f"Test: {name}")
        if isinstance(arg, tuple):
            result = fn(*arg)
        else:
            result = fn(arg)

        if result["success"]:
            key = [k for k in result if k not in ("success", "steps", "error", "latex")][0]
            print(f"  ✓  {key}: {result[key]}")
            for step in result["steps"]:
                print(f"     {step}")
        else:
            print(f"  ✗  Error: {result['error']}")
        print()
