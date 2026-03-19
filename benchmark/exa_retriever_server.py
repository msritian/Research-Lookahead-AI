"""
Exa Retriever Server
====================
A FastAPI microservice that exposes the same /retrieve endpoint
that Search-R1 expects, backed by the Exa live web search API.

Search-R1 calls:
    POST http://127.0.0.1:8000/retrieve
    {"queries": ["some question"], "topk": 3, "return_scores": true}

We forward each query to Exa, package results in Search-R1's expected
document format, and return them.

Run before benchmark:
    python benchmark/exa_retriever_server.py

Or from the Colab notebook (background thread).
"""

import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Request / Response schemas matching Search-R1's retriever contract
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True


class DocumentResult(BaseModel):
    document: dict       # {"contents": "\"Title\"\ntext"}
    score: float


class RetrieveResponse(BaseModel):
    result: List[List[DocumentResult]]


# ---------------------------------------------------------------------------
# Exa search logic
# ---------------------------------------------------------------------------

def exa_search(query: str, topk: int, api_key: str) -> List[DocumentResult]:
    try:
        from exa_py import Exa
        exa = Exa(api_key)

        response = exa.search_and_contents(
            query,
            num_results=topk,
            text=True,
            highlights=True,
        )

        docs = []
        for i, r in enumerate(response.results):
            title   = getattr(r, "title", "") or ""
            text    = getattr(r, "text", "")  or ""
            score   = getattr(r, "score", 1.0 - i * 0.05)

            # Search-R1 exact corpus format used in _passages2string():
            # first line = title, rest = body text
            # infer.py does: title = content.split("\n")[0]; text = "\n".join(content.split("\n")[1:])
            contents = f'"{title}"\n{text[:2000]}'

            docs.append(DocumentResult(
                document={"contents": contents},
                score=float(score),
            ))

        return docs

    except Exception as e:
        print(f"[ExaServer] Search error for query '{query[:60]}': {e}")
        return []


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Exa Retriever Server")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return RetrieveResponse(result=[[] for _ in request.queries])

    results = []
    for query in request.queries:
        docs = exa_search(query, topk=request.topk, api_key=api_key)
        results.append(docs)

    return RetrieveResponse(result=results)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "exa"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("RETRIEVER_PORT", 8000))
    print(f"[ExaServer] Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
