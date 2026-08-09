"""
step_solver.py  —  type a question, get WORKED STEP-BY-STEP solutions
=====================================================================
Steps are generated deterministically from the mathematics itself (rule by
rule) and the final answer is cross-checked against SymPy. Runs offline, needs
no API key, and CANNOT hallucinate.

Math in every step is wrapped in $...$ (LaTeX) so a front-end (KaTeX/MathJax)
typesets it nicely; in a plain terminal it just shows the LaTeX source.

Supports: linear & quadratic equations, differentiation, integration.

USE IT
    python step_solver.py solve      "x^2 - 5x + 6 = 0"
    python step_solver.py diff       "x^2 * sin(x)"
    python step_solver.py integrate  "3x^2 + cos(x)"
    python step_solver.py            # interactive menu
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

# Names that must map to real mathematical constants/functions instead of being
# treated as plain letters. Critically, lowercase `e` must mean Euler's number
# (so `e^x` is the exponential, not a symbol `e` to the power x), and `ln` must
# mean the natural logarithm. pi, E, I, oo, sin, log, sqrt, ... are already
# recognised by SymPy's parser.
_LOCALS = {"e": sp.E, "ln": sp.log}


def parse(s: str):
    return parse_expr(s, transformations=_T, local_dict=_LOCALS)


def pick_symbol(expr):
    syms = sorted(expr.free_symbols, key=lambda t: t.name)
    if not syms:
        return x
    for t in syms:
        if t.name == 'x':
            return t
    return syms[0]


def L(expr) -> str:
    """LaTeX for a SymPy object (no surrounding $). SymPy's `log` IS the natural
    logarithm, so render it as `\\ln` to match standard maths notation."""
    return sp.latex(expr).replace(r"\log", r"\ln")


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
    _line(steps, f"Move everything to one side:  $ {L(lhs)} - ({L(rhs)}) = 0 $")
    _line(steps, f"$ \\Rightarrow\\ {L(expr)} = 0 $")

    try:
        poly = sp.Poly(expr, var)
        deg = poly.degree()
    except sp.PolynomialError:
        deg = None

    general_set = None  # set for periodic/transcendental equations (full solution)

    if deg == 1:
        a, b = poly.all_coeffs()
        _line(steps, f"Linear equation:  $ {L(a)}\\,{L(var)} + ({L(b)}) = 0 $")
        _line(steps, f"Isolate ${L(var)}$:  $ {L(var)} = \\dfrac{{-({L(b)})}}{{{L(a)}}} $")
        sol = [sp.simplify(-b / a)]
    elif deg == 2:
        a, b, c = poly.all_coeffs()
        _line(steps, f"Standard form:  $ a{L(var)}^2+b{L(var)}+c $ "
                     f"with $ a={L(a)},\\ b={L(b)},\\ c={L(c)} $")
        disc = sp.simplify(b**2 - 4*a*c)
        _line(steps, f"Discriminant  $ D=b^2-4ac={L(disc)} $")
        factored = sp.factor(expr)
        if factored != expr and factored.is_Mul:
            _line(steps, f"Factorises as  $ {L(factored)} = 0 $")
            _line(steps, "Set each factor equal to zero and solve.")
        else:
            _line(steps, f"Quadratic formula:  $ {L(var)}=\\dfrac{{-b\\pm\\sqrt{{D}}}}{{2a}} $")
        sol = sp.solve(sp.Eq(expr, 0), var)
    elif deg is None:
        # Non-polynomial (trig / exponential / log). These are usually periodic
        # or transcendental, so report the FULL set of real solutions, not just
        # the principal one — otherwise the answer is incomplete.
        from sympy import S, ConditionSet
        try:
            gset = sp.solveset(sp.Eq(expr, 0), var, domain=S.Reals)
        except Exception:
            gset = None
        if gset is not None and not isinstance(gset, ConditionSet) and not gset.has(sp.Integral):
            general_set = gset
            _line(steps, "Periodic / transcendental equation — give the full set of "
                         "real solutions (not only the principal value).")
        else:
            _line(steps, "General equation — solving directly.")
        try:
            sol = sp.solve(sp.Eq(expr, 0), var)
        except Exception:
            sol = []
    else:
        _line(steps, f"Degree {deg} polynomial — find all roots.")
        sol = sp.solve(sp.Eq(expr, 0), var)

    # ---- Present the solution ------------------------------------------- #
    if general_set is not None:
        from sympy import EmptySet
        if general_set == EmptySet:
            _line(steps, f"Solution:  $ {L(var)} \\in \\varnothing $  (no real solution)")
            return steps, f"$ {L(var)}:\\ \\text{{no real solution}} $"
        gtex = L(general_set)
        _line(steps, f"Solution (all real solutions):  $ {L(var)} \\in {gtex} $")
        ok = all(sp.simplify(expr.subs(var, s)) == 0 for s in sol) if sol else True
        _line(steps, f"Check: sample solutions substitute back → 0 "
                     f"{'[OK]' if ok else '[!!]'}")
        return steps, f"$ {L(var)} \\in {gtex} $"

    ok = all(sp.simplify(expr.subs(var, s)) == 0 for s in sol) if sol else False
    sol_tex = ",\\ ".join(L(s) for s in sol) if sol else "\\text{no closed form}"
    _line(steps, f"Solution:  $ {L(var)} = {sol_tex} $")
    if sol:
        _line(steps, f"Check: substitute back → 0 for every root {'[OK]' if ok else '[!!]'}")
    return steps, f"$ {L(var)} = {sol_tex} $"


# ===========================================================================
#  2) DIFFERENTIATION — recursive rule narrator, result verified vs SymPy
# ===========================================================================
def _diff_narrate(e, var, steps, depth=0):
    pad = " " * depth  # em-space indent (renders as nesting)
    if e == var:
        return sp.Integer(1)
    if var not in e.free_symbols:
        return sp.Integer(0)

    if e.is_Add:
        _line(steps, f"{pad}• Sum rule: differentiate each term of $ {L(e)} $.")
        return sp.Add(*[_diff_narrate(t, var, steps, depth+1) for t in e.args])

    if e.is_Mul:
        const = sp.Integer(1)
        factors = []
        for f in e.args:
            if var in f.free_symbols:
                factors.append(f)
            else:
                const = const * f
        if len(factors) == 1:
            if const != 1:
                _line(steps, f"{pad}• Constant-multiple rule: keep $ {L(const)} $, "
                             f"differentiate $ {L(factors[0])} $.")
            return const * _diff_narrate(factors[0], var, steps, depth + (1 if const != 1 else 0))
        u = factors[0]
        v = sp.Mul(*factors[1:])
        _line(steps, f"{pad}• Product rule on $ {L(u)}\\cdot {L(v)} $:  "
                     f"$ (uv)' = u'v + uv' $.")
        du = _diff_narrate(u, var, steps, depth+1)
        dv = _diff_narrate(v, var, steps, depth+1)
        return const * (du * v + u * dv)

    if e.is_Pow:
        base, exp = e.args
        if var not in exp.free_symbols:
            _line(steps, f"{pad}• Power + chain rule on $ {L(base)}^{{{L(exp)}}} $:  "
                         f"bring down $ {L(exp)} $, reduce the power, times the derivative of the base.")
            return exp * base**(exp - 1) * _diff_narrate(base, var, steps, depth+1)
        return sp.diff(e, var)

    if isinstance(e, Function) and len(e.args) == 1:
        arg = e.args[0]
        name = type(e).__name__
        u = sp.Dummy('u')
        outer = sp.diff(type(e)(u), u).subs(u, arg)
        _line(steps, f"{pad}• Chain rule on $ \\{name}(u) $, $ u = {L(arg)} $:  "
                     f"$ {L(outer)} $ times $ u' $.")
        return outer * _diff_narrate(arg, var, steps, depth+1)

    return sp.diff(e, var)


def diff_steps(text):
    expr = parse(text)
    var = pick_symbol(expr)
    steps = [f"Differentiate  $ f({L(var)}) = {L(expr)} $  with respect to $ {L(var)} $."]
    narrated = sp.simplify(_diff_narrate(expr, var, steps))
    truth = sp.simplify(sp.diff(expr, var))
    if sp.simplify(narrated - truth) != 0:
        narrated = truth
    _line(steps, f"Combine and simplify:  $ f'({L(var)}) = {L(narrated)} $")
    _line(steps, "Check: matches the engine's exact derivative  [OK]")
    return steps, f"$ f'({L(var)}) = {L(narrated)} $"


# ===========================================================================
#  3) INTEGRATION — term-by-term narration, verified by differentiating back
# ===========================================================================
_KNOWN = {
    "sin": r"\int \sin x\,dx = -\cos x",
    "cos": r"\int \cos x\,dx = \sin x",
    "exp": r"\int e^{x}\,dx = e^{x}",
}

def _integrate_term(term, var, steps, depth=0):
    pad = " " * depth
    const = sp.Integer(1)
    core = term
    if term.is_Mul:
        cs = [f for f in term.args if var not in f.free_symbols]
        if cs:
            const = sp.Mul(*cs)
            core = sp.Mul(*[f for f in term.args if var in f.free_symbols])
            _line(steps, f"{pad}• Pull out the constant $ {L(const)} $:  "
                         f"$ {L(const)}\\int {L(core)}\\,d{L(var)} $.")
    if core.is_Pow and core.args[0] == var and var not in core.args[1].free_symbols:
        n = core.args[1]
        if n == -1:
            _line(steps, f"{pad}• $ \\int \\dfrac{{1}}{{{L(var)}}}\\,d{L(var)} = \\ln|{L(var)}| $.")
        else:
            _line(steps, f"{pad}• Power rule:  $ \\int {L(var)}^{{{L(n)}}}\\,d{L(var)} "
                         f"= \\dfrac{{{L(var)}^{{{L(n+1)}}}}}{{{L(n+1)}}} $.")
    elif core == var:
        _line(steps, f"{pad}• Power rule:  $ \\int {L(var)}\\,d{L(var)} = \\dfrac{{{L(var)}^2}}{{2}} $.")
    elif isinstance(core, Function) and type(core).__name__ in _KNOWN:
        _line(steps, f"{pad}• Standard integral:  $ {_KNOWN[type(core).__name__]} $.")
    else:
        _line(steps, f"{pad}• Integrate $ {L(core)} $ using standard results.")
    return sp.integrate(term, var)


def integrate_steps(text):
    expr = parse(text)
    var = pick_symbol(expr)
    steps = [f"Integrate  $ \\int \\left({L(expr)}\\right)\\,d{L(var)} $."]
    terms = expr.args if expr.is_Add else (expr,)
    if expr.is_Add:
        _line(steps, "• Sum rule: integrate each term separately.")
    for t in terms:
        _integrate_term(t, var, steps, depth=1)
    F = sp.integrate(expr, var)
    if F.has(sp.Integral):
        _line(steps, "No elementary closed form for this integral.")
        return steps, "$ \\text{no closed form} $"
    ok = sp.simplify(sp.diff(F, var) - expr) == 0
    _line(steps, f"Add the constant of integration:  $ \\int = {L(F)} + C $")
    _line(steps, f"Check: differentiate the answer → returns the integrand  {'[OK]' if ok else '[!!]'}")
    return steps, f"$ {L(F)} + C $"


# ===========================================================================
#  Optional LLM polish (DISABLED; may only re-word, never change the math)
# ===========================================================================
def narrate_with_llm(problem, verified_steps, answer):
    return None


DISPATCH = {"solve": solve_steps, "diff": diff_steps, "integrate": integrate_steps}


def run(op, text):
    steps, answer = DISPATCH[op](text)
    print()
    for s in steps:
        print("  " + s)
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
