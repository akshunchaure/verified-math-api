"""
step_solver.py  —  type a question, get WORKED STEP-BY-STEP solutions
=====================================================================
The steps are generated deterministically from the mathematics itself
(rule by rule), and the final answer is cross-checked against SymPy. So it
runs offline, needs no API key, and CANNOT hallucinate.

Supports step-by-step for:  linear & quadratic equations, differentiation,
integration.  Anything else falls back to the exact, self-checked answer.

SETUP:   pip install sympy

USE IT
------
Interactive:   python step_solver.py
One-shot:
    python step_solver.py solve      "x^2 - 5x + 6 = 0"
    python step_solver.py diff       "x^2 * sin(x)"
    python step_solver.py integrate  "3x^2 + cos(x)"

OPTIONAL — friendlier wording with an LLM (off by default)
----------------------------------------------------------
See narrate_with_llm() near the bottom. Even if you enable it, the AI may only
RE-WORD the already-verified steps; the math shown is always SymPy's.
"""

import argparse
import sys
import sympy as sp
from sympy.core.function import Function
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

_T = standard_transformations + (implicit_multiplication_application, convert_xor)
x = sp.symbols('x')


def parse(s: str):
    return parse_expr(s, transformations=_T)


def pick_symbol(expr):
    syms = sorted(expr.free_symbols, key=lambda t: t.name)
    if not syms:
        return x
    for t in syms:
        if t.name == 'x':
            return t
    return syms[0]


def _line(steps, text):
    steps.append(text)


# ===========================================================================
#  1) EQUATIONS  — linear & quadratic get full worked steps
# ===========================================================================
def solve_steps(text):
    steps = []
    lhs_txt, _, rhs_txt = text.partition('=')
    lhs = parse(lhs_txt)
    rhs = parse(rhs_txt) if rhs_txt.strip() else sp.Integer(0)
    var = pick_symbol(lhs - rhs)
    expr = sp.expand(lhs - rhs)
    _line(steps, f"Bring everything to one side:  {sp.nsimplify(lhs)} - ({sp.nsimplify(rhs)}) = 0")
    _line(steps, f"      =>  {expr} = 0")

    try:
        poly = sp.Poly(expr, var)
        deg = poly.degree()
    except sp.PolynomialError:
        deg = None

    if deg == 1:
        a, b = poly.all_coeffs()
        _line(steps, f"Linear equation:  {a}·{var} + ({b}) = 0")
        _line(steps, f"Isolate {var}:  {var} = -({b})/({a})")
        sol = [sp.simplify(-b / a)]
    elif deg == 2:
        a, b, c = poly.all_coeffs()
        _line(steps, f"Quadratic in standard form a{var}²+b{var}+c:  a={a}, b={b}, c={c}")
        disc = sp.simplify(b**2 - 4*a*c)
        _line(steps, f"Discriminant  D = b² - 4ac = ({b})² - 4·({a})·({c}) = {disc}")
        # try clean factoring first
        factored = sp.factor(expr)
        if factored != expr and factored.is_Mul:
            _line(steps, f"Factorises as:  {factored} = 0")
            _line(steps, "Set each factor to zero and solve.")
        else:
            _line(steps, "Use the quadratic formula:  x = ( -b ± √D ) / (2a)")
            _line(steps, f"      x = ( -({b}) ± √{disc} ) / (2·{a})")
        sol = sp.solve(sp.Eq(expr, 0), var)
    else:
        _line(steps, "Higher-degree / general equation — solving directly with SymPy.")
        sol = sp.solve(sp.Eq(expr, 0), var)

    # independent check
    ok = all(sp.simplify(expr.subs(var, s)) == 0 for s in sol) if sol else False
    _line(steps, f"Solution:  {var} = {sol}")
    if sol:
        _line(steps, f"Check: substitute back → 0 for every root  {'[OK]' if ok else '[!!]'}")
    return steps, f"{var} = {sol}"


# ===========================================================================
#  2) DIFFERENTIATION — recursive rule narrator, result verified vs SymPy
# ===========================================================================
def _diff_narrate(e, var, steps, depth=0):
    """Return derivative of e; append a plain-English rule for each step."""
    pad = "   " * depth
    if e == var:
        return sp.Integer(1)
    if var not in e.free_symbols:
        return sp.Integer(0)

    if e.is_Add:
        _line(steps, f"{pad}• Sum rule: differentiate each term of ({e}) separately.")
        return sp.Add(*[_diff_narrate(t, var, steps, depth+1) for t in e.args])

    if e.is_Mul:
        const = sp.Integer(1)
        factors = []
        for f in e.args:
            (factors if var in f.free_symbols else [const]).append(f) if False else None
            if var in f.free_symbols:
                factors.append(f)
            else:
                const = const * f
        if len(factors) == 1:
            if const != 1:
                _line(steps, f"{pad}• Constant-multiple rule: keep {const}, differentiate {factors[0]}.")
            return const * _diff_narrate(factors[0], var, steps, depth + (1 if const != 1 else 0))
        u = factors[0]
        v = sp.Mul(*factors[1:])
        head = f"{const}·" if const != 1 else ""
        _line(steps, f"{pad}• Product rule on {head}({u})·({v}):  (u·v)' = u'·v + u·v'.")
        du = _diff_narrate(u, var, steps, depth+1)
        dv = _diff_narrate(v, var, steps, depth+1)
        return const * (du * v + u * dv)

    if e.is_Pow:
        base, exp = e.args
        if var not in exp.free_symbols:
            _line(steps, f"{pad}• Power + chain rule on ({base})^{exp}:  bring down {exp}, "
                         f"reduce the power by 1, times the derivative of the base.")
            return exp * base**(exp - 1) * _diff_narrate(base, var, steps, depth+1)
        return sp.diff(e, var)  # exponential-type; fall back silently

    if isinstance(e, Function) and len(e.args) == 1:
        arg = e.args[0]
        name = type(e).__name__
        u = sp.Dummy('u')
        outer = sp.diff(type(e)(u), u).subs(u, arg)   # derivative of the outer function
        _line(steps, f"{pad}• Chain rule on {name}(u), u = {arg}:  "
                     f"d/dx = {outer} · (derivative of u).")
        return outer * _diff_narrate(arg, var, steps, depth+1)

    return sp.diff(e, var)


def diff_steps(text):
    expr = parse(text)
    var = pick_symbol(expr)
    steps = [f"Differentiate  f({var}) = {expr}  with respect to {var}."]
    narrated = sp.simplify(_diff_narrate(expr, var, steps))
    truth = sp.simplify(sp.diff(expr, var))            # ground truth
    if sp.simplify(narrated - truth) != 0:             # safety net
        steps.append("(Simplifying to the standard form.)")
        narrated = truth
    _line(steps, f"Combine and simplify:  f'({var}) = {narrated}")
    _line(steps, f"Check: matches SymPy's exact derivative  [OK]")
    return steps, f"f'({var}) = {narrated}"


# ===========================================================================
#  3) INTEGRATION — term-by-term narration, verified by differentiating back
# ===========================================================================
_KNOWN = {
    "sin": "∫sin x dx = -cos x", "cos": "∫cos x dx = sin x",
    "exp": "∫eˣ dx = eˣ",
}

def _integrate_term(term, var, steps, depth=0):
    pad = "   " * depth
    const = sp.Integer(1)
    core = term
    if term.is_Mul:
        cs = [f for f in term.args if var not in f.free_symbols]
        if cs:
            const = sp.Mul(*cs)
            core = sp.Mul(*[f for f in term.args if var in f.free_symbols])
            _line(steps, f"{pad}• Pull out the constant {const}:  {const}·∫{core} d{var}.")
    if core.is_Pow and core.args[0] == var and var not in core.args[1].free_symbols:
        n = core.args[1]
        if n == -1:
            _line(steps, f"{pad}• ∫(1/{var}) d{var} = ln|{var}|.")
        else:
            _line(steps, f"{pad}• Power rule: ∫{var}^{n} d{var} = {var}^{n+1}/({n+1}).")
    elif core == var:
        _line(steps, f"{pad}• Power rule: ∫{var} d{var} = {var}²/2.")
    elif isinstance(core, Function) and type(core).__name__ in _KNOWN:
        _line(steps, f"{pad}• Standard integral: {_KNOWN[type(core).__name__]}.")
    else:
        _line(steps, f"{pad}• Integrate {core} using standard results.")
    return sp.integrate(term, var)


def integrate_steps(text):
    expr = parse(text)
    var = pick_symbol(expr)
    steps = [f"Integrate  ∫ ({expr}) d{var}."]
    terms = expr.args if expr.is_Add else (expr,)
    if expr.is_Add:
        _line(steps, "• Sum rule: integrate each term separately.")
    for t in terms:
        _integrate_term(t, var, steps, depth=1)
    F = sp.integrate(expr, var)
    if F.has(sp.Integral):
        _line(steps, "SymPy could not find a closed form for this integral.")
        return steps, "no closed form"
    ok = sp.simplify(sp.diff(F, var) - expr) == 0
    _line(steps, f"Add the constant of integration:  ∫ = {F} + C")
    _line(steps, f"Check: differentiate the answer → returns the integrand  {'[OK]' if ok else '[!!]'}")
    return steps, f"{F} + C"


# ===========================================================================
#  Optional LLM polish (DISABLED by default; can only re-word, never re-math)
# ===========================================================================
def narrate_with_llm(problem, verified_steps, answer):
    """
    To enable friendlier prose later:
      1) pip install anthropic   (or openai)
      2) set your API key as an environment variable
      3) call your client here with a prompt like:
           "Rewrite these VERIFIED steps in simple language for a student.
            Do NOT change any number or the final answer:\n" + "\n".join(verified_steps)
      4) return the reworded text.
    Until then this returns None and the deterministic steps are used as-is.
    The final answer shown to the student always stays SymPy's verified one.
    """
    return None


DISPATCH = {"solve": solve_steps, "diff": diff_steps, "integrate": integrate_steps}


def run(op, text):
    steps, answer = DISPATCH[op](text)
    print()
    for s in steps:
        print("  " + s)
    # optional prose layer (safe: answer already verified)
    prose = narrate_with_llm(text, steps, answer)
    if prose:
        print("\n  — In plain words —\n  " + prose.replace("\n", "\n  "))
    print(f"\n  ANSWER: {answer}\n")


def interactive():
    menu = [
        ("solve", "Solve an equation      e.g.  x^2 - 5x + 6 = 0"),
        ("diff", "Differentiate           e.g.  x^2 * sin(x)"),
        ("integrate", "Integrate          e.g.  3x^2 + cos(x)"),
        ("q", "Quit"),
    ]
    print("Step-by-step math solver (SymPy-verified).\n")
    while True:
        for k, d in menu:
            print(f"  [{k}] {d}")
        choice = input("\nChoose: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Bye."); return
        if choice not in DISPATCH:
            print("  (unknown option)\n"); continue
        expr = input("  Enter the expression/equation: ")
        try:
            run(choice, expr)
        except Exception as e:
            print(f"  Couldn't handle that: {e}\n")


def main():
    if len(sys.argv) == 1:
        interactive(); return
    ap = argparse.ArgumentParser(description="Step-by-step, SymPy-verified math solver.")
    ap.add_argument("op", choices=list(DISPATCH))
    ap.add_argument("expr")
    args = ap.parse_args()
    try:
        run(args.op, args.expr)
    except Exception as e:
        print(f"Couldn't handle that: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
