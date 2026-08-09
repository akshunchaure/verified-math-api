"""
app.py  —  Verified Math API (FastAPI backend)
==============================================
Exposes the SymPy engine as a small JSON web API so a front-end (e.g. the
Lovable app) can use it. The math stays in SymPy, so answers cannot hallucinate.

It returns LaTeX strings (not PDFs), so the server needs NO LaTeX install —
the front-end renders the LaTeX with KaTeX. That keeps deployment trivial.

RUN LOCALLY:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    # open http://localhost:8000/docs  for an interactive API explorer

ENDPOINTS
    GET  /health                      -> status + chapter list
    GET  /chapters                    -> list of generator chapters
    POST /steps    {op, expr}         -> worked step-by-step + verified answer
    POST /generate {chapter,count,difficulty,seed} -> verified problem set
"""

import json
import os
import random
import urllib.request
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import mathpack_generator as mp
import step_solver as ss

# --- Optional Gemini layer (plain-English explanations of VERIFIED steps) ---
# Set GEMINI_API_KEY (and optionally GEMINI_MODEL) as environment variables on
# the server. The AI only RE-WORDS steps SymPy already verified; it never
# changes a number or the answer. With no key, /explain still returns the
# verified steps + answer (explanation = null).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def _gemini_explain(problem, steps, answer):
    if not GEMINI_API_KEY:
        return None
    prompt = (
        "You are a warm, encouraging maths teacher. The steps below have ALREADY "
        "been verified as correct by a computer algebra system. Rewrite them as a "
        "short, clear plain-English explanation a student can follow (4-8 sentences). "
        "Do NOT change any number, expression, or the final answer, and do not add "
        "new steps.\n\n"
        f"Problem: {problem}\n\nVerified steps:\n" + "\n".join(steps) +
        f"\n\nVerified final answer: {answer}\n\nExplanation:"
    )
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Exam-level ("JEE") generation: the AI PROPOSES a hard question, SymPy
#  INDEPENDENTLY solves + verifies it. The student only ever sees SymPy's
#  verified answer, never the AI's unchecked one — the whole point of the app.
# --------------------------------------------------------------------------- #
def _gemini_generate_problem(topic, difficulty):
    """Ask Gemini for ONE hard JEE-style problem that reduces to a SymPy-checkable
    operation (solve/diff/integrate). Returns the raw model text (JSON) or None."""
    if not GEMINI_API_KEY:
        return None
    prompt = (
        "You are an expert JEE (India, Class 11-12) mathematics problem setter. "
        f"Create ONE original, challenging problem on the topic \"{topic}\" at difficulty "
        f"{difficulty} out of 3 (3 = JEE Advanced level, like the harder RD Sharma / NCERT "
        "exemplar problems). It MUST be solvable by exactly ONE of these operations so a "
        "computer algebra system can verify the answer: 'solve' (solve an equation for x), "
        "'diff' (differentiate an expression in x), or 'integrate' (integrate an expression "
        "in x). Return STRICT JSON ONLY (no markdown fences, no commentary) with exactly "
        "these keys: {\"problem\": \"<the question a student reads>\", "
        "\"op\": \"solve|diff|integrate\", \"expr\": \"<equation/expression in plain "
        "SymPy/Python syntax>\"}. Rules: use ** or ^ for powers, * for multiplication, and "
        "sin, cos, tan, cot, sec, csc, log, ln, sqrt, exp, e (Euler's number), pi. For "
        "'solve', expr MUST contain '=' (e.g. 'x**2 - 5*x + 6 = 0'). Make difficulty-3 "
        "problems genuinely hard (messy integrals needing substitution or parts; higher-"
        "degree, parameterised, or trigonometric equations; product/quotient/chain-rule-"
        "heavy derivatives). The 'problem' text and the 'expr' must describe the SAME thing. "
        "FORMATTING of the 'problem' text: write EVERY mathematical symbol or expression as "
        "inline LaTeX wrapped in single dollar signs, e.g. $z^3 = 2\\bar{z}^2$, $\\int e^x "
        "\\sin x\\,dx$, $x^{10}$. Use ^ with braces for multi-character exponents ($x^{2n}$), "
        "\\frac for fractions, \\bar, \\sqrt, \\pi, etc. NEVER use backticks, code formatting, "
        "or markdown in the 'problem' text. (This LaTeX rule is ONLY for 'problem'; 'expr' must "
        "stay plain SymPy syntax with ** or ^.) "
        "Return only the JSON object."
    )
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 1.0}}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


def _parse_problem_json(raw):
    """Pull the JSON object out of the model text, tolerating ``` code fences."""
    if not raw:
        return None
    t = raw.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1 and t[:nl].strip().lower() in ("json", ""):
            t = t[nl + 1:]
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except Exception:
        return None
    if not all(k in obj for k in ("problem", "op", "expr")):
        return None
    if obj["op"] not in ("solve", "diff", "integrate"):
        return None
    return obj


app = FastAPI(title="Verified Math API", version="1.0",
              description="SymPy-verified math: worked solutions + infinite practice.")

# Allow the Lovable front-end (any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your app's domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Health / metadata
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "engine": "sympy", "chapters": list(mp.CHAPTERS),
            "ai_explanations": bool(GEMINI_API_KEY)}


@app.get("/chapters")
def chapters():
    return {"chapters": list(mp.CHAPTERS) + ["mixed"],
            "difficulty": {1: "foundational", 2: "moderate", 3: "advanced"}}


# --------------------------------------------------------------------------- #
#  Step-by-step solver
# --------------------------------------------------------------------------- #
class StepReq(BaseModel):
    op: str = Field(..., description="solve | diff | integrate")
    expr: str = Field(..., description="e.g. 'x^2 - 5x + 6 = 0' or 'x*sin(x)'")


@app.post("/steps")
def steps(req: StepReq):
    fn = ss.DISPATCH.get(req.op)
    if fn is None:
        raise HTTPException(400, f"op must be one of {list(ss.DISPATCH)}")
    try:
        step_list, answer = fn(req.expr)
    except Exception as e:
        raise HTTPException(422, f"Could not process that expression: {e}")
    return {"op": req.op, "input": req.expr, "steps": step_list, "answer": answer}


# --------------------------------------------------------------------------- #
#  Plain-English explanation (Gemini re-words the VERIFIED steps)
# --------------------------------------------------------------------------- #
class ExplainReq(BaseModel):
    op: str = Field(..., description="solve | diff | integrate")
    expr: str = Field(..., description="the expression/equation")


@app.post("/explain")
def explain(req: ExplainReq):
    fn = ss.DISPATCH.get(req.op)
    if fn is None:
        raise HTTPException(400, f"op must be one of {list(ss.DISPATCH)}")
    try:
        step_list, answer = fn(req.expr)          # SymPy-verified steps + answer
    except Exception as e:
        raise HTTPException(422, f"Could not process that expression: {e}")
    explanation = _gemini_explain(req.expr, step_list, answer)
    return {
        "op": req.op, "input": req.expr,
        "steps": step_list, "answer": answer,
        "explanation": explanation,               # null if no key / call failed
        "ai_enabled": bool(GEMINI_API_KEY),
    }


# --------------------------------------------------------------------------- #
#  Plain-English explanation for a PRACTICE ITEM (already-generated problem)
#  The practice generator produces problems with their own verified solution
#  steps, so we ask Gemini to re-word THOSE steps (it never changes the maths).
# --------------------------------------------------------------------------- #
class ExplainItemReq(BaseModel):
    problem: str = Field(..., description="the practice problem (may contain LaTeX)")
    solution: List[str] = Field(default_factory=list, description="the verified solution steps")
    answer: str = Field("", description="the final verified answer")


@app.post("/explain_item")
def explain_item(req: ExplainItemReq):
    explanation = _gemini_explain(req.problem, req.solution, req.answer)
    return {
        "explanation": explanation,               # null if no key / call failed
        "ai_enabled": bool(GEMINI_API_KEY),
    }


# --------------------------------------------------------------------------- #
#  Exam-level (JEE / Class 11-12) question — AI-proposed, SymPy-verified
# --------------------------------------------------------------------------- #
class ExamReq(BaseModel):
    chapter: str = Field("mixed", description="a topic name (flavour for the AI)")
    difficulty: int = Field(3, ge=1, le=3, description="1-3; 3 = JEE Advanced level")


@app.post("/exam")
def exam(req: ExamReq):
    if not GEMINI_API_KEY:
        raise HTTPException(503, "Exam-level generation needs an AI key on the server.")
    valid = set(mp.CHAPTERS) | {"mixed"}
    topic = req.chapter if req.chapter in valid else "mixed"
    last_err = None
    for _ in range(5):                       # generate -> verify -> retry if not checkable
        obj = _parse_problem_json(_gemini_generate_problem(topic, req.difficulty))
        if obj is None:
            last_err = "model did not return a valid problem"
            continue
        try:
            # SymPy INDEPENDENTLY solves the proposed question; this is the verification.
            steps, answer = ss.DISPATCH[obj["op"]](obj["expr"])
        except Exception as e:
            last_err = f"unverifiable expression: {e}"
            continue
        # reject degenerate results so students always get a real, solvable question
        if "no closed form" in answer or "no real solution" in answer:
            last_err = "generated a degenerate problem"
            continue
        return {
            "chapter": topic,
            "difficulty": req.difficulty,
            "problem": obj["problem"],
            "op": obj["op"],
            "expr": obj["expr"],
            "steps": steps,                  # SymPy-verified worked solution
            "answer": answer,                # SymPy-verified answer (never the AI's)
            "verified": True,
        }
    raise HTTPException(502, f"Couldn't generate a verified question — please try again. "
                            f"({last_err or 'unknown'})")


# --------------------------------------------------------------------------- #
#  Verified problem generator
# --------------------------------------------------------------------------- #
class GenReq(BaseModel):
    chapter: str = Field("mixed", description="a chapter name or 'mixed'")
    count: int = Field(5, ge=1, le=15, description="problems (per chapter if mixed)")
    difficulty: int = Field(2, ge=1, le=3)
    seed: Optional[int] = Field(None, description="omit for a fresh set each call")


@app.post("/generate")
def generate(req: GenReq):
    valid = set(mp.CHAPTERS) | {"mixed"}
    if req.chapter not in valid:
        raise HTTPException(400, f"chapter must be one of {sorted(valid)}")
    rng = random.Random(req.seed)
    try:
        items = mp.build_pack(req.chapter, req.count, req.difficulty, rng)
    except Exception as e:
        raise HTTPException(422, f"Generation failed: {e}")
    # independent audit before returning (never ship unverified items)
    if not all(mp.verify(it) for it in items):
        raise HTTPException(500, "internal verification failed")
    return {
        "chapter": req.chapter,
        "difficulty": req.difficulty,
        "count": len(items),
        "items": [
            {
                "chapter": it.chapter,
                "problem": it.problem_latex,
                "answer": it.answer_latex,
                "solution": it.solution_latex,
                "check": it.verify_note,
            }
            for it in items
        ],
    }


@app.get("/")
def root():
    return {"message": "Verified Math API is running. See /docs for usage."}
