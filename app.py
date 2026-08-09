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

import random
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import mathpack_generator as mp
import step_solver as ss

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
    return {"status": "ok", "engine": "sympy", "chapters": list(mp.CHAPTERS)}


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
