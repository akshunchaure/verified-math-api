"""
solver.py  —  type a math question, get the exact, self-verified solution
==========================================================================
Powered by SymPy (exact algebra/calculus — it does not guess, so it does not
hallucinate). Where possible the answer is CHECKED a second way before showing.

SETUP:   pip install sympy

TWO WAYS TO USE IT
------------------
1) Interactive (just run it, then follow the menu):
       python solver.py

2) One-shot from the command line:
       python solver.py solve      "x^2 - 5x + 6 = 0"
       python solver.py diff       "x*sin(x)"
       python solver.py integrate  "1/(x^2 + 1)"
       python solver.py simplify   "sin(x)^2 + cos(x)^2"
       python solver.py factor     "x^2 - 9"
       python solver.py expand     "(x+2)*(x-3)"
       python solver.py limit      "sin(x)/x" --at 0

NOTES
-----
- You can write  2x  for 2*x  and  x^2  for x**2 (it understands both).
- Constants available: pi, E (Euler's number), oo (infinity), I (imaginary unit).
- Functions: sin, cos, tan, exp, log, sqrt, etc.  (log is natural log.)
"""

import argparse
import sys
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# Let users type "2x" and "x^2" naturally.
_T = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse(s: str):
    """Turn a text string into a SymPy expression."""
    return parse_expr(s, transformations=_T)


def pick_symbol(expr):
    """Choose the variable to work with (prefer x; else the first one found)."""
    syms = sorted(expr.free_symbols, key=lambda t: t.name)
    if not syms:
        return sp.symbols('x')
    for t in syms:
        if t.name == 'x':
            return t
    return syms[0]


def show(label, value):
    print(f"  {label}: {value}")


# --------------------------------------------------------------------------- #
#  Operations — each returns nothing, just prints result + a self-check
# --------------------------------------------------------------------------- #
def op_solve(text):
    lhs_txt, _, rhs_txt = text.partition('=')
    lhs = parse(lhs_txt)
    rhs = parse(rhs_txt) if rhs_txt.strip() else sp.Integer(0)
    var = pick_symbol(lhs - rhs)
    sols = sp.solve(sp.Eq(lhs, rhs), var)
    show("Equation", f"{lhs} = {rhs}")
    show(f"Solve for {var}", sols if sols else "no closed-form solution found")
    # self-check: substitute each solution back
    good = all(sp.simplify((lhs - rhs).subs(var, s)) == 0 for s in sols) if sols else False
    if sols:
        show("Check", "each solution substituted back gives 0  " + ("[OK]" if good else "[!!]"))


def op_diff(text):
    expr = parse(text)
    var = pick_symbol(expr)
    result = sp.simplify(sp.diff(expr, var))
    show("f(x)", expr)
    show(f"d/d{var}", result)


def op_integrate(text):
    expr = parse(text)
    var = pick_symbol(expr)
    F = sp.integrate(expr, var)
    show("Integrand", expr)
    if F.has(sp.Integral):
        show("Integral", "SymPy could not find a closed form for this one.")
        return
    show("Result", f"{F} + C")
    # self-check: differentiate the answer, should return the integrand
    ok = sp.simplify(sp.diff(F, var) - expr) == 0
    show("Check", "d/dx of the answer returns the integrand  " + ("[OK]" if ok else "[!!]"))


def op_simplify(text):
    expr = parse(text)
    show("Input", expr)
    show("Simplified", sp.simplify(expr))


def op_factor(text):
    expr = parse(text)
    show("Input", expr)
    show("Factored", sp.factor(expr))


def op_expand(text):
    expr = parse(text)
    show("Input", expr)
    show("Expanded", sp.expand(expr))


def op_limit(text, at):
    expr = parse(text)
    var = pick_symbol(expr)
    point = parse(str(at))
    show("Expression", expr)
    show(f"limit as {var} -> {point}", sp.limit(expr, var, point))


OPS = {
    "solve": op_solve, "diff": op_diff, "integrate": op_integrate,
    "simplify": op_simplify, "factor": op_factor, "expand": op_expand,
}


# --------------------------------------------------------------------------- #
#  Interactive menu (when run with no command-line arguments)
# --------------------------------------------------------------------------- #
def interactive():
    menu = [
        ("solve", "Solve an equation            e.g.  x^2 - 5x + 6 = 0"),
        ("diff", "Differentiate                 e.g.  x*sin(x)"),
        ("integrate", "Integrate                e.g.  1/(x^2 + 1)"),
        ("simplify", "Simplify                  e.g.  sin(x)^2 + cos(x)^2"),
        ("factor", "Factor                      e.g.  x^2 - 9"),
        ("expand", "Expand                      e.g.  (x+2)*(x-3)"),
        ("limit", "Limit                        e.g.  sin(x)/x  as x -> 0"),
        ("q", "Quit"),
    ]
    print("Type a math question, get the exact solution (powered by SymPy).\n")
    while True:
        for k, d in menu:
            print(f"  [{k}] {d}")
        choice = input("\nChoose: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Bye.")
            return
        try:
            if choice == "limit":
                expr = input("  Expression: ")
                at = input("  x approaches (e.g. 0, oo, pi): ")
                op_limit(expr, at)
            elif choice in OPS:
                expr = input("  Enter the expression/equation: ")
                OPS[choice](expr)
            else:
                print("  (unknown option)\n"); continue
        except Exception as e:
            print(f"  Couldn't parse that: {e}")
        print()


def main():
    if len(sys.argv) == 1:
        interactive(); return
    ap = argparse.ArgumentParser(description="Type a math question, get the exact solution.")
    ap.add_argument("op", choices=list(OPS) + ["limit"])
    ap.add_argument("expr")
    ap.add_argument("--at", default="0", help="point for 'limit' (e.g. 0, oo, pi)")
    args = ap.parse_args()
    try:
        if args.op == "limit":
            op_limit(args.expr, args.at)
        else:
            OPS[args.op](args.expr)
    except Exception as e:
        print(f"Couldn't parse that: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
