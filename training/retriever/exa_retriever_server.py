"""
Exa Retriever Server — Training Version
=========================================
Same /retrieve endpoint contract as Search-R1's retrieval_server.py,
backed by Exa live web search instead of a local corpus.

Key addition over the benchmark version: supports end_published_date
so the rollout engine can enforce temporal grounding during training
(only fetch articles published before the market's resolution date).

Search-R1's rollout calls:
    POST http://127.0.0.1:8000/retrieve
    {"queries": ["..."], "topk": 3, "return_scores": true}

We extend the request schema with an optional end_date field:
    {"queries": ["..."], "topk": 3, "end_date": "2025-03-01"}

If end_date is provided, Exa only returns articles published before it.

Run:
    EXA_API_KEY=xxx python training/retriever/exa_retriever_server.py
"""

import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True
    end_date: Optional[str] = None   # YYYY-MM-DD — articles published before this date only


class DocumentResult(BaseModel):
    document: dict   # {"contents": "\"Title\"\ntext", "published_date": "2025-02-10"}
    score: float


class RetrieveResponse(BaseModel):
    result: List[List[DocumentResult]]


def exa_search(
    query: str,
    topk: int,
    api_key: str,
    end_date: Optional[str] = None,
) -> List[DocumentResult]:
    try:
        from exa_py import Exa
        exa = Exa(api_key)

        kwargs = dict(
            num_results=topk,
            text=True,
            highlights=True,
        )
        if end_date:
            kwargs["end_published_date"] = f"{end_date}T00:00:00Z"

        response = exa.search_and_contents(query, **kwargs)

        docs = []
        for i, r in enumerate(response.results):
            title          = getattr(r, "title",          "") or ""
            text           = getattr(r, "text",           "") or ""
            score          = getattr(r, "score",          None)
            published_date = getattr(r, "published_date", None)  # "2025-02-10T..."

            score = float(score) if score is not None else 1.0 - i * 0.05

            # Normalise to YYYY-MM-DD for reward scoring
            pub_date_str = None
            if published_date:
                pub_date_str = str(published_date)[:10]

            contents = f'"{title}"\n{text[:2000]}'
            docs.append(DocumentResult(
                document={
                    "contents":       contents,
                    "published_date": pub_date_str,   # NEW — used by temporal reward
                },
                score=score,
            ))

        return docs

    except Exception as e:
        print(f"[ExaServer] Search error for '{query[:60]}': {e}")
        return []


app = FastAPI(title="Exa Retriever Server (Training)")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        print("[ExaServer] WARNING: EXA_API_KEY not set — returning empty results")
        return RetrieveResponse(result=[[] for _ in request.queries])

    results = []
    for query in request.queries:
        docs = exa_search(
            query,
            topk=request.topk,
            api_key=api_key,
            end_date=request.end_date,
        )
        results.append(docs)

    return RetrieveResponse(result=results)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "exa"}


if __name__ == "__main__":
    port = int(os.environ.get("RETRIEVER_PORT", 8000))
    print(f"[ExaServer] Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
