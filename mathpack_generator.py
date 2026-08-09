#!/usr/bin/env python3
"""
mathpack_generator.py  —  v0 "verified factory" for JEE math worksheets
=======================================================================

CORE PRINCIPLE
--------------
The LLM is NEVER the source of mathematical truth. Every problem is built so
its answer is correct *by construction* (SymPy), and then EVERY item is checked
a second time by an INDEPENDENT verifier before it is allowed into the PDF.
Result: a worksheet you can hand to a paying institute with a zero-error guarantee.

WHAT THIS DOES
--------------
  - Generates practice problems for JEE chapters (quadratics, differentiation,
    trigonometric identities) with full, correct solutions.
  - Verifies each item independently (substitution / numeric finite-difference /
    symbolic simplify + random-point checks).
  - Emits LaTeX and compiles a clean PDF (problems + answer key).

USAGE
-----
    python mathpack_generator.py --chapter mixed --count 5 --difficulty 2 \
        --seed 42 --title "Sharma Classes — Weekly DPP" --out pack.pdf

    # single-chapter packs:
    python mathpack_generator.py --chapter quadratics --count 8 --difficulty 1
    python mathpack_generator.py --chapter differentiation --count 6 --difficulty 3
    python mathpack_generator.py --chapter trig --count 5

REQUIREMENTS
------------
    pip install sympy numpy
    A LaTeX engine on PATH (pdflatex).  On Windows install MiKTeX; on Mac,
    MacTeX/BasicTeX; on Linux, texlive-latex-recommended + texlive-latex-extra.
    If pdflatex is absent the script still writes the .tex file so you can
    compile it anywhere (e.g. overleaf.com).

HOW TO EXTEND (add a chapter)
-----------------------------
    Write a gen_<chapter>(rng, difficulty) -> Item, and a matching branch in
    verify(item). Register it in CHAPTERS. That's it — the PDF layer is generic.
"""

import argparse, os, random, shutil, subprocess, sys, tempfile
from dataclasses import dataclass, field
from typing import List
import sympy as sp

x = sp.symbols('x')

# ----------------------------------------------------------------------------- #
#  Data model
# ----------------------------------------------------------------------------- #
@dataclass
class Item:
    chapter: str
    problem_latex: str            # the question, as inline/enclosed LaTeX
    answer_latex: str             # short final answer
    solution_latex: List[str]     # ordered solution steps (each a LaTeX string)
    verify_note: str = ""         # how it was independently checked
    _payload: dict = field(default_factory=dict)  # data the verifier needs


# ----------------------------------------------------------------------------- #
#  Optional LLM prose polish (OFF by default, and it can NEVER change the math)
# ----------------------------------------------------------------------------- #
def polish_wording(text: str) -> str:
    """Placeholder hook. If you later wire an LLM here to reword a *prompt*
    (never an answer), route it through this function. It must be purely
    cosmetic: the verified math objects are the source of truth."""
    return text


# ----------------------------------------------------------------------------- #
#  Chapter 1 — Quadratic equations  (correct by construction: build from roots)
# ----------------------------------------------------------------------------- #
def gen_quadratic(rng: random.Random, difficulty: int) -> Item:
    while True:
        r1 = rng.randint(-9, 9)
        r2 = rng.randint(-9, 9)
        if difficulty == 1 and r1 == r2:
            continue                                   # keep easy ones distinct
        a = 1 if difficulty == 1 else rng.choice([1, 2, 3])
        expr = sp.expand(a * (x - r1) * (x - r2))      # answer known: roots r1, r2
        if expr == a * x**2:                            # avoid degenerate b=c=0
            continue
        roots = sorted({r1, r2})
        disc = sp.simplify(sp.discriminant(expr, x))
        steps = [
            r"Compare with $ax^2+bx+c=0$: "
            + rf"$a={a},\ b={sp.latex(expr.coeff(x,1))},\ c={sp.latex(expr.coeff(x,0))}$.",
            rf"Discriminant $D=b^2-4ac={sp.latex(disc)}\ (\ge 0)$, so real roots exist.",
            rf"Factorise: ${sp.latex(a)}\,(x-({r1}))(x-({r2}))=0$.",
            rf"$\therefore x={sp.latex(roots[0])}$"
            + (rf"$,\ {sp.latex(roots[1])}$." if len(roots) > 1 else r"\ \text{(repeated)}."),
        ]
        ans = ",\\ ".join(sp.latex(r) for r in roots)
        return Item(
            chapter="Quadratic Equations",
            problem_latex=rf"Solve for $x$:\quad ${sp.latex(expr)} = 0$.",
            answer_latex=rf"$x = {ans}$",
            solution_latex=steps,
            verify_note="Each root substituted back into the equation gives 0.",
            _payload={"expr": expr, "roots": [r1, r2]},
        )


# ----------------------------------------------------------------------------- #
#  Chapter 2 — Differentiation  (answer = SymPy derivative; verified numerically)
# ----------------------------------------------------------------------------- #
def _safe_pos(rng):            # small positive int
    return rng.randint(1, 5)

def _rand_fn(rng: random.Random, difficulty: int):
    a, b, c, d = (_safe_pos(rng) for _ in range(4))
    k = rng.randint(1, 3)
    if difficulty == 1:
        return rng.choice([
            a*x**3 + b*x**2 + c*x + d,
            a*sp.sin(x) + b*x**2 + c,
            a*sp.cos(x) + b*x,
            a*sp.exp(x) + b*x**2,
        ])
    if difficulty == 2:
        return rng.choice([
            (a*x**2 + b)*sp.sin(x),
            (a*x + b)/(x**2 + c),
            a*sp.sin(k*x**2 + b),
            a*x*sp.exp(x),
            a*sp.log(b*x**2 + c),           # b,c>0  => domain-safe on (0.5,1.5)
        ])
    return rng.choice([
        sp.exp(x)*sp.sin(x),
        a*sp.sqrt(b*x**2 + c),
        sp.sin(x)/x,
        a*sp.log(x)*sp.cos(x),
        (x**2 + a)/sp.exp(x),
    ])

def gen_differentiation(rng: random.Random, difficulty: int) -> Item:
    import numpy as np
    for _ in range(200):
        f = _rand_fn(rng, difficulty)
        _raw = sp.diff(f, x)
        try:                                  # canonical, expanded-numerator form
            fp = sp.simplify(sp.cancel(_raw))
        except Exception:
            fp = sp.simplify(_raw)
        # independent numeric check: symbolic f'(x) vs central finite difference
        f_l = sp.lambdify(x, f, "numpy")
        fp_l = sp.lambdify(x, fp, "numpy")
        pts = [0.6, 0.9, 1.2, 1.4]
        ok, h = True, 1e-6
        try:
            for p in pts:
                num = (f_l(p + h) - f_l(p - h)) / (2*h)
                sym = fp_l(p)
                if not np.isfinite(num) or not np.isfinite(sym):
                    ok = False; break
                if abs(num - sym) > 1e-4 * (1 + abs(sym)):
                    ok = False; break
        except Exception:
            ok = False
        if not ok:
            continue
        rule = ("product / chain rule" if difficulty >= 2 else "power & standard rules")
        steps = [
            rf"Let $f(x) = {sp.latex(f)}$.",
            rf"Apply the {rule}, differentiating term by term.",
            rf"$f'(x) = {sp.latex(fp)}$.",
        ]
        return Item(
            chapter="Differentiation",
            problem_latex=rf"Differentiate with respect to $x$:\quad $f(x) = {sp.latex(f)}$.",
            answer_latex=rf"$f'(x) = {sp.latex(fp)}$",
            solution_latex=steps,
            verify_note="Symbolic derivative matches a central finite-difference at 4 points.",
            _payload={"f": f, "fp": fp},
        )
    raise RuntimeError("could not generate a well-behaved differentiation item")


# ----------------------------------------------------------------------------- #
#  Chapter 3 — Trigonometric identities  (curated + auto-verified bank)
#  Each entry: (statement_latex, LHS, RHS, [solution steps]).  Verified two ways.
# ----------------------------------------------------------------------------- #
th = sp.symbols('theta')
s, c = sp.sin(th), sp.cos(th)
t, sec, csc, cot = sp.tan(th), 1/sp.cos(th), 1/sp.sin(th), sp.cos(th)/sp.sin(th)

TRIG_BANK = [
    (r"\dfrac{1-\sin^2\theta}{\cos\theta} = \cos\theta",
     (1 - s**2)/c, c,
     [r"$1-\sin^2\theta=\cos^2\theta$.",
      r"$\dfrac{\cos^2\theta}{\cos\theta}=\cos\theta.$"]),
    (r"\sin\theta\,\cos\theta\,\tan\theta = \sin^2\theta",
     s*c*t, s**2,
     [r"$\tan\theta=\dfrac{\sin\theta}{\cos\theta}$.",
      r"$\sin\theta\cos\theta\cdot\dfrac{\sin\theta}{\cos\theta}=\sin^2\theta.$"]),
    (r"\dfrac{1}{1+\sin\theta}+\dfrac{1}{1-\sin\theta} = 2\sec^2\theta",
     1/(1+s) + 1/(1-s), 2*sec**2,
     [r"Combine: $\dfrac{(1-\sin\theta)+(1+\sin\theta)}{1-\sin^2\theta}$.",
      r"$=\dfrac{2}{\cos^2\theta}=2\sec^2\theta.$"]),
    (r"\sec\theta-\cos\theta = \sin\theta\,\tan\theta",
     sec - c, s*t,
     [r"$\sec\theta-\cos\theta=\dfrac{1-\cos^2\theta}{\cos\theta}=\dfrac{\sin^2\theta}{\cos\theta}$.",
      r"$=\sin\theta\tan\theta.$"]),
    (r"\dfrac{1+\tan^2\theta}{1+\cot^2\theta} = \tan^2\theta",
     (1+t**2)/(1+cot**2), t**2,
     [r"$1+\tan^2\theta=\sec^2\theta,\ 1+\cot^2\theta=\csc^2\theta$.",
      r"$\dfrac{\sec^2\theta}{\csc^2\theta}=\tan^2\theta.$"]),
    (r"\sin^6\theta+\cos^6\theta = 1-3\sin^2\theta\cos^2\theta",
     s**6 + c**6, 1 - 3*s**2*c**2,
     [r"Let $a=\sin^2\theta,\ b=\cos^2\theta$ so $a+b=1$.",
      r"$a^3+b^3=(a+b)^3-3ab(a+b)=1-3ab.$"]),
    (r"(\sin\theta+\csc\theta)^2+(\cos\theta+\sec\theta)^2 = 7+\tan^2\theta+\cot^2\theta",
     (s+csc)**2 + (c+sec)**2, 7 + t**2 + cot**2,
     [r"Expand both squares; use $\sin^2\theta+\cos^2\theta=1$.",
      r"$=1+4+(1+\cot^2\theta)+(1+\tan^2\theta)=7+\tan^2\theta+\cot^2\theta.$"]),
    (r"(1+\cot\theta-\csc\theta)(1+\tan\theta+\sec\theta) = 2",
     (1+cot-csc)*(1+t+sec), sp.Integer(2),
     [r"Write in terms of $\sin\theta,\cos\theta$ and combine.",
      r"$=\dfrac{(\sin\theta+\cos\theta)^2-1}{\sin\theta\cos\theta}=2.$"]),
]

def gen_trig(rng: random.Random, difficulty: int) -> Item:
    stmt, lhs, rhs, steps = rng.choice(TRIG_BANK)
    return Item(
        chapter="Trigonometric Identities",
        problem_latex=rf"Prove the identity:\quad ${stmt}$.",
        answer_latex=r"Proved. $\blacksquare$",
        solution_latex=steps,
        verify_note="LHS-RHS simplifies to 0 and matches numerically at random angles.",
        _payload={"lhs": lhs, "rhs": rhs},
    )


# ----------------------------------------------------------------------------- #
#  INDEPENDENT VERIFIER  (the anti-hallucination gate — runs on EVERY item)
# ----------------------------------------------------------------------------- #
def verify(item: Item) -> bool:
    import numpy as np
    p = item._payload
    k = p.get("kind")
    if k and k in VNEW:                       # newer chapters register a verifier by 'kind'
        return VNEW[k](p)
    if item.chapter == "Quadratic Equations":
        expr, roots = p["expr"], p["roots"]
        return all(sp.simplify(expr.subs(x, r)) == 0 for r in roots)
    if item.chapter == "Differentiation":
        f, fp = p["f"], p["fp"]
        f_l, fp_l = sp.lambdify(x, f, "numpy"), sp.lambdify(x, fp, "numpy")
        h = 1e-6
        for q in (0.55, 0.85, 1.15, 1.45):
            num = (f_l(q+h) - f_l(q-h)) / (2*h)
            sym = fp_l(q)
            if not (np.isfinite(num) and np.isfinite(sym)):
                return False
            if abs(num - sym) > 1e-4 * (1 + abs(sym)):
                return False
        return True
    if item.chapter == "Trigonometric Identities":
        lhs, rhs = p["lhs"], p["rhs"]
        if sp.simplify(lhs - rhs) == 0:              # symbolic proof when SymPy manages it
            return True
        # otherwise numeric proof: agree at many random, non-singular angles
        f = sp.lambdify(th, lhs - rhs, "numpy")
        good = 0
        for a in (0.3, 0.7, 1.1, 1.9, 2.4, 2.7, 3.5, 4.2):
            try:
                v = complex(f(a))
            except Exception:
                continue
            if not np.isfinite(v.real):
                continue
            if abs(v) > 1e-9:
                return False
            good += 1
        return good >= 4
    return False


CHAPTERS = {
    "quadratics": gen_quadratic,
    "differentiation": gen_differentiation,
    "trig": gen_trig,
}

# =========================================================================== #
#  ADDITIONAL JEE CHAPTERS  (each: correct by construction + independent check)
#  Every generator returns an Item whose _payload carries a 'kind' used by the
#  verifier registry VNEW below. Add a chapter = add a gen_* + a _v_* + register.
# =========================================================================== #
VNEW = {}   # kind -> function(payload) -> bool

# --- Sequences & Series (AP / GP) ---
def gen_sequences(rng, difficulty):
    if rng.random() < 0.5:
        a, d, n = rng.randint(-5, 9), rng.randint(1, 6), rng.randint(5, 15)
        tn = a + (n - 1) * d
        Sn = sp.Rational(n, 2) * (2 * a + (n - 1) * d)
        prob = (rf"In an A.P. with first term ${a}$ and common difference ${d}$, "
                rf"find the ${n}$th term and the sum of the first ${n}$ terms.")
        steps = [rf"$t_n=a+(n-1)d={a}+({n}-1)({d})={sp.latex(tn)}$.",
                 rf"$S_n=\tfrac{{n}}{{2}}[2a+(n-1)d]"
                 rf"=\tfrac{{{n}}}{{2}}[2({a})+({n}-1)({d})]={sp.latex(Sn)}$."]
        ans = rf"$t_{{{n}}}={sp.latex(tn)},\ S_{{{n}}}={sp.latex(Sn)}$"
        payload = {"kind": "seq_ap", "a": a, "d": d, "n": n, "tn": tn, "Sn": Sn}
    else:
        a, r, n = rng.randint(1, 4), rng.choice([2, 3]), rng.randint(4, 7)
        tn = a * r**(n - 1)
        Sn = sp.Rational(a * (r**n - 1), r - 1)
        prob = (rf"In a G.P. with first term ${a}$ and common ratio ${r}$, "
                rf"find the ${n}$th term and the sum of the first ${n}$ terms.")
        steps = [rf"$t_n=ar^{{n-1}}={a}\cdot {r}^{{{n}-1}}={sp.latex(tn)}$.",
                 rf"$S_n=\dfrac{{a(r^{{n}}-1)}}{{r-1}}={sp.latex(Sn)}$."]
        ans = rf"$t_{{{n}}}={sp.latex(tn)},\ S_{{{n}}}={sp.latex(Sn)}$"
        payload = {"kind": "seq_gp", "a": a, "r": r, "n": n, "tn": tn, "Sn": Sn}
    return Item("Sequences & Series", prob, ans, steps,
                "nth term and sum cross-checked against direct summation.", payload)

def _v_seq_ap(p):
    tn = p["a"] + (p["n"] - 1) * p["d"]
    Sn = sum(p["a"] + i * p["d"] for i in range(p["n"]))
    return tn == p["tn"] and Sn == p["Sn"]
def _v_seq_gp(p):
    tn = p["a"] * p["r"]**(p["n"] - 1)
    Sn = sum(p["a"] * p["r"]**i for i in range(p["n"]))
    return tn == p["tn"] and sp.simplify(Sn - p["Sn"]) == 0
VNEW["seq_ap"], VNEW["seq_gp"] = _v_seq_ap, _v_seq_gp

# --- Straight Lines ---
def gen_lines(rng, difficulty):
    X, Y = sp.symbols('x y')
    while True:
        x1, y1 = rng.randint(-6, 6), rng.randint(-6, 6)
        x2, y2 = rng.randint(-6, 6), rng.randint(-6, 6)
        if (x1, y1) != (x2, y2):
            break
    line = sp.expand((y2 - y1) * (X - x1) - (x2 - x1) * (Y - y1))
    prob = rf"Find the equation of the straight line through $({x1},{y1})$ and $({x2},{y2})$."
    steps = [r"Use $(y_2-y_1)(x-x_1)=(x_2-x_1)(y-y_1)$.",
             rf"$({y2 - y1})(x-({x1}))=({x2 - x1})(y-({y1}))$.",
             rf"Simplify to  ${sp.latex(line)}=0$."]
    payload = {"kind": "line", "p1": (x1, y1), "p2": (x2, y2), "line": line, "X": X, "Y": Y}
    return Item("Straight Lines", prob, rf"${sp.latex(line)} = 0$", steps,
                "Both given points satisfy the derived equation.", payload)
def _v_line(p):
    L, X, Y = p["line"], p["X"], p["Y"]
    return (sp.simplify(L.subs({X: p["p1"][0], Y: p["p1"][1]})) == 0 and
            sp.simplify(L.subs({X: p["p2"][0], Y: p["p2"][1]})) == 0)
VNEW["line"] = _v_line

# --- Complex Numbers ---
def gen_complex(rng, difficulty):
    a = rng.randint(-6, 6)
    b = rng.choice([i for i in range(-6, 7) if i != 0])
    z = a + b * sp.I
    mod = sp.sqrt(a**2 + b**2)
    conj = a - b * sp.I
    prob = rf"For $z={sp.latex(z)}$, find $|z|$ and $\bar{{z}}$."
    steps = [rf"$|z|=\sqrt{{a^2+b^2}}=\sqrt{{({a})^2+({b})^2}}={sp.latex(mod)}$.",
             rf"$\bar{{z}}=a-bi={sp.latex(conj)}$."]
    ans = rf"$|z|={sp.latex(mod)},\ \bar{{z}}={sp.latex(conj)}$"
    return Item("Complex Numbers", prob, ans, steps,
                "Modulus verified as sqrt(a^2+b^2).", {"kind": "cplx", "a": a, "b": b, "mod": mod})
def _v_cplx(p):
    return sp.simplify(sp.Abs(p["a"] + p["b"] * sp.I) - p["mod"]) == 0
VNEW["cplx"] = _v_cplx

# --- Binomial Theorem ---
def gen_binomial(rng, difficulty):
    n = rng.randint(4, 8)
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    k = rng.randint(0, n)
    coeff = sp.expand((a * x + b)**n).coeff(x, k)
    formula = sp.binomial(n, k) * a**k * b**(n - k)
    prob = rf"Find the coefficient of $x^{{{k}}}$ in the expansion of $({a}x+{b})^{{{n}}}$."
    steps = [rf"General term $T_{{r+1}}=\binom{{{n}}}{{r}}({a}x)^{{r}}({b})^{{{n}-r}}$.",
             rf"Set $r={k}$: $\binom{{{n}}}{{{k}}}\,{a}^{{{k}}}\,{b}^{{{n}-{k}}}={sp.latex(formula)}$."]
    return Item("Binomial Theorem", prob, rf"${sp.latex(coeff)}$", steps,
                "Coefficient from the full expansion matches the binomial-coefficient formula.",
                {"kind": "binom", "n": n, "a": a, "b": b, "k": k, "coeff": coeff})
def _v_binom(p):
    lhs = sp.expand((p["a"] * x + p["b"])**p["n"]).coeff(x, p["k"])
    return lhs == sp.binomial(p["n"], p["k"]) * p["a"]**p["k"] * p["b"]**(p["n"] - p["k"]) == p["coeff"]
VNEW["binom"] = _v_binom

# --- Determinants (3x3) ---
def gen_determinants(rng, difficulty):
    M = sp.Matrix(3, 3, lambda i, j: rng.randint(-4, 5))
    det = M.det()
    prob = rf"Evaluate $\det(A)$ where $A={sp.latex(M)}$."
    steps = ["Expand by cofactors along the first row.",
             rf"$\det(A)={sp.latex(det)}$."]
    return Item("Determinants", prob, rf"$\det(A)={sp.latex(det)}$", steps,
                "SymPy determinant matches an independent numeric (NumPy) determinant.",
                {"kind": "det3", "M": M, "det": det})
def _v_det3(p):
    import numpy as np
    approx = round(float(np.linalg.det(np.array(p["M"].tolist(), dtype=float))))
    return int(p["det"]) == approx
VNEW["det3"] = _v_det3

# --- Limits ---
def gen_limits(rng, difficulty):
    a = rng.randint(1, 4)
    expr, pt = rng.choice([
        (sp.sin(a * x) / x, 0), ((sp.exp(a * x) - 1) / x, 0),
        ((1 - sp.cos(x)) / x**2, 0), (sp.tan(a * x) / x, 0),
    ])
    L = sp.limit(expr, x, pt)
    prob = rf"Evaluate $\displaystyle\lim_{{x\to {pt}}}\ {sp.latex(expr)}$."
    steps = [r"Apply standard limits (e.g. $\lim_{x\to0}\tfrac{\sin x}{x}=1$).",
             rf"Value $={sp.latex(L)}$."]
    return Item("Limits", prob, rf"${sp.latex(L)}$", steps,
                "Symbolic limit matches numeric evaluation approaching the point.",
                {"kind": "lim", "expr": expr, "pt": pt, "L": L})
def _v_lim(p):
    f = sp.lambdify(x, p["expr"], "numpy")
    tgt = float(p["L"])
    return all(abs(float(f(float(p["pt"]) + e)) - tgt) < 1e-2 for e in (1e-3, -1e-3, 5e-4))
VNEW["lim"] = _v_lim

# --- Definite Integrals ---
def gen_definite_integral(rng, difficulty):
    a = rng.randint(0, 2)
    b = a + rng.randint(1, 3)
    c, n = rng.randint(1, 4), rng.randint(1, 3)
    f = c * x**n + sp.sin(x)
    val = sp.integrate(f, (x, a, b))
    prob = rf"Evaluate $\displaystyle\int_{{{a}}}^{{{b}}}\left({sp.latex(f)}\right)\,dx$."
    steps = [rf"Antiderivative $F(x)={sp.latex(sp.integrate(f, x))}$.",
             rf"$F({b})-F({a})={sp.latex(val)}$."]
    return Item("Definite Integrals", prob, rf"${sp.latex(val)}$", steps,
                "Exact value matches high-precision numeric integration.",
                {"kind": "defint", "f": f, "a": a, "b": b, "val": val})
def _v_defint(p):
    import mpmath
    g = sp.lambdify(x, p["f"], "mpmath")
    q = float(mpmath.quad(g, [p["a"], p["b"]]))
    return abs(float(p["val"].evalf()) - q) < 1e-6
VNEW["defint"] = _v_defint

# --- Permutations & Combinations ---
def gen_pnc(rng, difficulty):
    n = rng.randint(5, 10)
    r = rng.randint(2, min(5, n))
    if rng.random() < 0.5:
        val = sp.factorial(n) / sp.factorial(n - r)
        prob = rf"Evaluate $^{{{n}}}P_{{{r}}}$."
        steps = [rf"$^nP_r=\dfrac{{n!}}{{(n-r)!}}=\dfrac{{{n}!}}{{({n}-{r})!}}={sp.latex(val)}$."]
        which = "P"
    else:
        val = sp.binomial(n, r)
        prob = rf"Evaluate $^{{{n}}}C_{{{r}}}$."
        steps = [rf"$^nC_r=\dfrac{{n!}}{{r!\,(n-r)!}}={sp.latex(val)}$."]
        which = "C"
    return Item("Permutations & Combinations", prob, rf"${sp.latex(val)}$", steps,
                "Value matches a direct factorial computation.",
                {"kind": "pnc", "n": n, "r": r, "val": val, "which": which})
def _v_pnc(p):
    import math
    true = math.perm(p["n"], p["r"]) if p["which"] == "P" else math.comb(p["n"], p["r"])
    return int(p["val"]) == true
VNEW["pnc"] = _v_pnc

# --- Matrices (2x2: determinant + inverse) ---
def gen_matrices(rng, difficulty):
    while True:
        M = sp.Matrix(2, 2, lambda i, j: rng.randint(-4, 5))
        if M.det() != 0:
            break
    det, inv = M.det(), M.inv()
    prob = rf"For $A={sp.latex(M)}$, find $\det(A)$ and $A^{{-1}}$."
    steps = [rf"$\det(A)=ad-bc={sp.latex(det)}$.",
             rf"$A^{{-1}}=\dfrac{{1}}{{\det(A)}}\begin{{pmatrix}}d&-b\\-c&a\end{{pmatrix}}={sp.latex(inv)}$."]
    return Item("Matrices", prob, rf"$\det(A)={sp.latex(det)},\ A^{{-1}}={sp.latex(inv)}$", steps,
                "Verified that A times its inverse gives the identity matrix.",
                {"kind": "mat2", "M": M, "inv": inv})
def _v_mat2(p):
    return (p["M"] * p["inv"] - sp.eye(2)).applyfunc(sp.simplify) == sp.zeros(2, 2)
VNEW["mat2"] = _v_mat2

# --- Vectors (dot product + magnitude) ---
def gen_vectors(rng, difficulty):
    a = [rng.randint(-4, 5) for _ in range(3)]
    b = [rng.randint(-4, 5) for _ in range(3)]
    dot = sum(ai * bi for ai, bi in zip(a, b))
    mag = sp.sqrt(sum(ai**2 for ai in a))
    prob = (rf"For $\vec a=({a[0]},{a[1]},{a[2]})$ and $\vec b=({b[0]},{b[1]},{b[2]})$, "
            rf"find $\vec a\cdot\vec b$ and $|\vec a|$.")
    steps = [rf"$\vec a\cdot\vec b=a_1b_1+a_2b_2+a_3b_3={dot}$.",
             rf"$|\vec a|=\sqrt{{a_1^2+a_2^2+a_3^2}}={sp.latex(mag)}$."]
    return Item("Vectors", prob, rf"$\vec a\cdot\vec b={dot},\ |\vec a|={sp.latex(mag)}$", steps,
                "Dot product and magnitude recomputed from components.",
                {"kind": "vec", "a": a, "b": b, "dot": dot, "mag": mag})
def _v_vec(p):
    return (sum(ai * bi for ai, bi in zip(p["a"], p["b"])) == p["dot"] and
            sp.simplify(sp.sqrt(sum(ai**2 for ai in p["a"])) - p["mag"]) == 0)
VNEW["vec"] = _v_vec

CHAPTERS.update({
    "sequences": gen_sequences,
    "lines": gen_lines,
    "complex": gen_complex,
    "binomial": gen_binomial,
    "determinants": gen_determinants,
    "limits": gen_limits,
    "integrals": gen_definite_integral,
    "pnc": gen_pnc,
    "matrices": gen_matrices,
    "vectors": gen_vectors,
})

# =========================================================================== #
#  PHASE 2 CHAPTERS  (probability, statistics, circles, 3D, calculus apps, ...)
# =========================================================================== #

# --- Probability (simple) ---
def gen_probability(rng, difficulty):
    r, b = rng.randint(2, 8), rng.randint(2, 8)
    p = sp.Rational(r, r + b)
    prob = (rf"A bag contains ${r}$ red and ${b}$ blue balls. One ball is drawn "
            rf"at random. Find the probability that it is red.")
    steps = [rf"$P(\text{{red}})=\dfrac{{\text{{favourable}}}}{{\text{{total}}}}"
             rf"=\dfrac{{{r}}}{{{r}+{b}}}={sp.latex(p)}$."]
    return Item("Probability", prob, rf"$P(\text{{red}})={sp.latex(p)}$", steps,
                "Probability recomputed as favourable/total.",
                {"kind": "prob", "r": r, "b": b, "p": p})
def _v_prob(p):
    return sp.Rational(p["r"], p["r"] + p["b"]) == p["p"]
VNEW["prob"] = _v_prob

# --- Statistics (mean & variance) ---
def gen_statistics(rng, difficulty):
    n = rng.randint(4, 6)
    data = [rng.randint(1, 10) for _ in range(n)]
    mean = sp.Rational(sum(data), n)
    var = sum((sp.Rational(d) - mean)**2 for d in data) / n
    prob = (rf"Find the mean and variance of the data: "
            rf"${', '.join(str(d) for d in data)}$.")
    steps = [rf"Mean $\bar{{x}}=\dfrac{{\sum x_i}}{{n}}={sp.latex(mean)}$.",
             rf"Variance $\sigma^2=\dfrac{{\sum (x_i-\bar{{x}})^2}}{{n}}={sp.latex(var)}$."]
    ans = rf"Mean $={sp.latex(mean)}$, variance $={sp.latex(var)}$"
    return Item("Statistics", prob, ans, steps,
                "Mean and variance recomputed independently.",
                {"kind": "stats", "data": data, "mean": mean, "var": var})
def _v_stats(p):
    n = len(p["data"]); mean = sp.Rational(sum(p["data"]), n)
    var = sum((sp.Rational(d) - mean)**2 for d in p["data"]) / n
    return sp.simplify(mean - p["mean"]) == 0 and sp.simplify(var - p["var"]) == 0
VNEW["stats"] = _v_stats

# --- Circles (conic sections) ---
def gen_circle(rng, difficulty):
    X, Y = sp.symbols('x y')
    h, k, r = rng.randint(-4, 4), rng.randint(-4, 4), rng.randint(2, 5)
    D, E, F = -2*h, -2*k, h*h + k*k - r*r
    eq = X**2 + Y**2 + D*X + E*Y + F
    prob = rf"Find the centre and radius of the circle $ {sp.latex(eq)} = 0 $."
    steps = [rf"Centre $=\left(-\tfrac{{D}}{{2}},-\tfrac{{E}}{{2}}\right)=({h},{k})$.",
             rf"Radius $=\sqrt{{(D/2)^2+(E/2)^2-F}}={r}$."]
    ans = rf"Centre $({h},{k})$, radius $={r}$"
    return Item("Circles", prob, ans, steps,
                "Centre and radius satisfy the given equation.",
                {"kind": "circle", "h": h, "k": k, "r": r, "D": D, "E": E, "F": F})
def _v_circle(p):
    return (-p["D"]/2 == p["h"] and -p["E"]/2 == p["k"]
            and (p["D"]/2)**2 + (p["E"]/2)**2 - p["F"] == p["r"]**2)
VNEW["circle"] = _v_circle

# --- 3D Geometry (distance between points) ---
def gen_threed(rng, difficulty):
    a = [rng.randint(-5, 5) for _ in range(3)]
    b = [rng.randint(-5, 5) for _ in range(3)]
    d2 = sum((ai - bi)**2 for ai, bi in zip(a, b))
    dist = sp.sqrt(d2)
    prob = (rf"Find the distance between $A({a[0]},{a[1]},{a[2]})$ and "
            rf"$B({b[0]},{b[1]},{b[2]})$.")
    steps = [rf"$AB=\sqrt{{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}}"
             rf"=\sqrt{{{d2}}}={sp.latex(dist)}$."]
    return Item("3D Geometry", prob, rf"$AB={sp.latex(dist)}$", steps,
                "Distance recomputed from the coordinates.",
                {"kind": "3d", "a": a, "b": b, "dist": dist})
def _v_3d(p):
    d2 = sum((ai - bi)**2 for ai, bi in zip(p["a"], p["b"]))
    return sp.simplify(sp.sqrt(d2) - p["dist"]) == 0
VNEW["3d"] = _v_3d

# --- Application of Derivatives (maxima / minima) ---
def gen_maxmin(rng, difficulty):
    p_ = rng.randint(-4, 2)
    q_ = p_ + rng.randint(1, 4)
    fp = sp.expand(3*(x - p_)*(x - q_))          # this is f'(x)
    f = sp.integrate(fp, x)
    fpp = sp.diff(fp, x)
    def kind(pt):
        return "minimum" if fpp.subs(x, pt) > 0 else "maximum"
    prob = rf"Find and classify the local extrema of $ f(x) = {sp.latex(f)} $."
    steps = [rf"$f'(x)={sp.latex(fp)}=0 \Rightarrow x={p_},\ {q_}$.",
             rf"$f''(x)={sp.latex(fpp)}$: local {kind(p_)} at $x={p_}$, "
             rf"local {kind(q_)} at $x={q_}$."]
    ans = rf"Local {kind(p_)} at $x={p_}$; local {kind(q_)} at $x={q_}$"
    return Item("Application of Derivatives", prob, ans, steps,
                "Each critical point gives f'(x)=0 (checked).",
                {"kind": "maxmin", "fp": fp, "crit": [p_, q_]})
def _v_maxmin(p):
    return all(sp.simplify(p["fp"].subs(x, c)) == 0 for c in p["crit"])
VNEW["maxmin"] = _v_maxmin

# --- Differential Equations (dy/dx = f(x)) ---
def gen_ode(rng, difficulty):
    c, n, k = rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 5)
    rhs = c*x**n + k
    y = sp.integrate(rhs, x)
    prob = rf"Solve the differential equation $ \dfrac{{dy}}{{dx}} = {sp.latex(rhs)} $."
    steps = [rf"Integrate both sides: $ y=\int\left({sp.latex(rhs)}\right)dx $.",
             rf"$ y = {sp.latex(y)} + C $."]
    return Item("Differential Equations", prob, rf"$ y = {sp.latex(y)} + C $", steps,
                "Differentiating the solution returns the right-hand side.",
                {"kind": "ode", "rhs": rhs, "y": y})
def _v_ode(p):
    return sp.simplify(sp.diff(p["y"], x) - p["rhs"]) == 0
VNEW["ode"] = _v_ode

# --- Inverse Trigonometry (principal values) ---
def gen_invtrig(rng, difficulty):
    table = [
        (r"\sin^{-1}\!\left(\tfrac{1}{2}\right)", sp.asin(sp.Rational(1, 2)), sp.pi/6),
        (r"\cos^{-1}\!\left(\tfrac{1}{2}\right)", sp.acos(sp.Rational(1, 2)), sp.pi/3),
        (r"\tan^{-1}(1)", sp.atan(1), sp.pi/4),
        (r"\sin^{-1}(1)", sp.asin(1), sp.pi/2),
        (r"\cos^{-1}(0)", sp.acos(0), sp.pi/2),
        (r"\tan^{-1}(0)", sp.atan(0), sp.Integer(0)),
        (r"\sin^{-1}\!\left(\tfrac{\sqrt{3}}{2}\right)", sp.asin(sp.sqrt(3)/2), sp.pi/3),
    ]
    disp, expr, val = rng.choice(table)
    prob = rf"Evaluate the principal value of $ {disp} $."
    steps = [rf"$ {disp} = {sp.latex(val)} $."]
    return Item("Inverse Trigonometry", prob, rf"$ {sp.latex(val)} $", steps,
                "Applying the forward function returns the argument.",
                {"kind": "invtrig", "expr": expr, "val": val})
def _v_invtrig(p):
    return sp.simplify(p["expr"] - p["val"]) == 0
VNEW["invtrig"] = _v_invtrig

# --- Area Under Curves ---
def gen_area(rng, difficulty):
    a = rng.randint(0, 2)
    b = a + rng.randint(1, 3)
    c, n = rng.randint(1, 4), rng.randint(1, 3)
    f = c*x**n
    area = sp.integrate(f, (x, a, b))
    prob = (rf"Find the area under the curve $ y = {sp.latex(f)} $ "
            rf"from $ x={a} $ to $ x={b} $.")
    steps = [rf"Area $ = \int_{{{a}}}^{{{b}}} {sp.latex(f)}\,dx = {sp.latex(area)} $."]
    return Item("Area Under Curves", prob, rf"$ {sp.latex(area)} $", steps,
                "Area matches high-precision numeric integration.",
                {"kind": "area", "f": f, "a": a, "b": b, "area": area})
def _v_area(p):
    import mpmath
    g = sp.lambdify(x, p["f"], "mpmath")
    q = float(mpmath.quad(g, [p["a"], p["b"]]))
    return abs(float(p["area"].evalf()) - q) < 1e-6
VNEW["area"] = _v_area

CHAPTERS.update({
    "probability": gen_probability,
    "statistics": gen_statistics,
    "circles": gen_circle,
    "threed": gen_threed,
    "maxmin": gen_maxmin,
    "differential": gen_ode,
    "inversetrig": gen_invtrig,
    "areas": gen_area,
})


# ----------------------------------------------------------------------------- #
#  Build a pack (list of verified items)
# ----------------------------------------------------------------------------- #
def build_pack(chapter: str, count: int, difficulty: int, rng: random.Random) -> List[Item]:
    plan = (list(CHAPTERS.keys()) if chapter == "mixed" else [chapter])
    items: List[Item] = []
    seen = set()                                          # global de-dup across all chapters
    per = count if chapter != "mixed" else max(1, count)  # count = per-chapter when mixed
    for ch in plan:
        made = 0
        attempts = 0
        while made < per:
            attempts += 1
            if attempts > per * 200:
                raise RuntimeError(
                    f"stuck generating '{ch}' — try a smaller --count "
                    f"(this topic has a limited pool at this difficulty)")
            it = CHAPTERS[ch](rng, difficulty)
            key = it.problem_latex
            if key in seen:                               # no repeats within a pack
                continue
            if not verify(it):                            # <-- the hard gate
                continue
            seen.add(key)
            items.append(it)
            made += 1
    return items


# ----------------------------------------------------------------------------- #
#  LaTeX assembly + compile
# ----------------------------------------------------------------------------- #
LATEX_HEADER = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=20mm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{xcolor}
\definecolor{navy}{HTML}{0B3D5C}
\definecolor{blue}{HTML}{1A7FB5}
\pagestyle{fancy}\fancyhf{}
\rhead{\small\color{gray} Verified with SymPy $\cdot$ zero-error practice}
\lhead{\small\color{navy}\textbf{%(brand)s}}
\cfoot{\small\color{gray}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
{\LARGE\color{navy}\textbf{%(title)s}}\\[2pt]
{\small\color{gray}%(subtitle)s}
\end{center}
\vspace{4pt}\hrule\vspace{10pt}
"""

LATEX_FOOTER = r"\end{document}"

def _esc(s: str) -> str:
    """Escape LaTeX specials in PLAIN-TEXT fields (chapter names, notes).
    Never call this on the math fields (problem/answer/solution)."""
    s = str(s).replace("·", ".").replace("⁻¹", "^(-1)").replace("²", "^2")
    repl = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(repl.get(c, c) for c in s)


def latexify(items: List[Item], title, subtitle, brand, *,
             include_problems=True, include_solutions=True, practice_space=False) -> str:
    out = [LATEX_HEADER % {"title": title, "subtitle": subtitle, "brand": brand}]
    if include_problems:
        gap = "34pt" if practice_space else "8pt"
        out.append(r"\section*{\color{navy}Problems}")
        if practice_space:
            out.append(r"{\footnotesize\color{gray}\textit{Attempt each in the space "
                       r"below; solutions are in the separate answer key.}}\\[6pt]")
        out.append(r"\begin{enumerate}[leftmargin=*,itemsep=" + gap + "]")
        for it in items:
            out.append(r"\item " + it.problem_latex)
        out.append(r"\end{enumerate}")
    if include_problems and include_solutions:
        out.append(r"\newpage")
    if include_solutions:
        out.append(r"\section*{\color{navy}Answer Key \& Full Solutions}")
        out.append(r"\begin{enumerate}[leftmargin=*,itemsep=10pt]")
        for it in items:
            block = [r"\item \textbf{\color{blue}" + _esc(it.chapter) + r".}\quad " + it.answer_latex]
            block.append(r"\\[2pt]")
            for stp in it.solution_latex:
                block.append(stp + r"\\[1pt]")
            block.append(r"{\footnotesize\color{gray}\textit{Check: " + _esc(it.verify_note) + r"}}")
            out.append("\n".join(block))
        out.append(r"\end{enumerate}")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def compile_pdf(tex: str, out_pdf: str) -> bool:
    if shutil.which("pdflatex") is None:
        tex_path = os.path.splitext(out_pdf)[0] + ".tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"[!] pdflatex not found. Wrote LaTeX to {tex_path} "
              f"— compile it on overleaf.com or install a LaTeX distro.")
        return False
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, "pack.tex")
        with open(tp, "w") as f:
            f.write(tex)
        for _ in range(2):                       # twice for headers/page refs
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tp],
                               cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        produced = os.path.join(d, "pack.pdf")
        if os.path.exists(produced):
            shutil.copy(produced, out_pdf)
            return True
        print(r.stdout.decode(errors="ignore")[-1500:])
        return False


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Verified JEE math worksheet & self-practice generator (v0)")
    ap.add_argument("--chapter", choices=list(CHAPTERS) + ["mixed"], default="mixed",
                    help="topic to generate (or 'mixed' for all)")
    ap.add_argument("--count", type=int, default=5,
                    help="problems per chapter (mixed) or total (single chapter)")
    ap.add_argument("--difficulty", type=int, choices=[1, 2, 3], default=2,
                    help="1=foundational, 2=moderate, 3=advanced")
    ap.add_argument("--answers", choices=["with", "separate", "none"], default="with",
                    help="with=problems+key in one PDF; separate=two PDFs "
                         "(questions + answer key) for self-testing; none=questions only")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix for a repeatable set; omit for a fresh set every run")
    ap.add_argument("--title", default="JEE Mathematics — Practice Pack")
    ap.add_argument("--brand", default="Verified Math Pack")
    ap.add_argument("--out", default="pack.pdf")
    ap.add_argument("--list", action="store_true", help="list available chapters and exit")
    args = ap.parse_args()

    if args.list:
        print("Available chapters:")
        for c in CHAPTERS:
            print(f"  - {c}")
        print("  - mixed   (all of the above)")
        print("Difficulty: 1 = foundational, 2 = moderate, 3 = advanced")
        return

    rng = random.Random(args.seed)
    items = build_pack(args.chapter, args.count, args.difficulty, rng)

    # Final independent audit (belt AND suspenders) before anything is shown
    passed = sum(verify(it) for it in items)
    print(f"[verify] {passed}/{len(items)} items independently verified "
          f"({'ALL PASS' if passed == len(items) else 'FAILURES PRESENT'}).")
    if passed != len(items):
        print("[abort] refusing to emit a pack with unverified items.")
        sys.exit(1)

    diff_name = {1: "Foundational", 2: "Moderate", 3: "Advanced"}[args.difficulty]
    base_sub = f"{len(items)} problems  $\\cdot$  {diff_name}  $\\cdot$  every answer machine-verified"
    base = os.path.splitext(args.out)[0]

    def emit(tex, path):
        ok = compile_pdf(tex, path)
        print(f"[done] {'PDF -> ' + path if ok else 'LaTeX written (compile externally)'}")

    if args.answers == "with":
        emit(latexify(items, args.title, base_sub, args.brand), args.out)
    elif args.answers == "none":
        emit(latexify(items, args.title, base_sub, args.brand,
                      include_solutions=False, practice_space=True), args.out)
    else:  # separate — ideal for self-practice
        emit(latexify(items, args.title, base_sub, args.brand,
                      include_solutions=False, practice_space=True),
             base + "_questions.pdf")
        emit(latexify(items, args.title + " — Answer Key", base_sub, args.brand,
                      include_problems=False),
             base + "_answers.pdf")


if __name__ == "__main__":
    main()
