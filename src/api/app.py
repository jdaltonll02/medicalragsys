"""
FastAPI application for Medical RAG System.

Two route groups:
  /api/v1/*  — mobile contract  (Flutter app)
  /*         — legacy internal  (tooling / backward compat)

Authentication
--------------
The Flutter app sends  Authorization: Bearer <Firebase-JWT>  on every request.
This server currently accepts any bearer token without validation so the
mobile app works without a separate auth backend.  To enforce auth later,
uncomment the firebase-admin verification block in _get_optional_token().
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import yaml
from fastapi import FastAPI, HTTPException, APIRouter, Request, Header
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    # Mobile schemas
    MobileQueryRequest, MobileQueryResponse, SourceItem, HistoryPage, ErrorResponse,
    # Legacy schemas
    QueryRequest, QueryResponse, Document,
)
from src.pipeline.med_rag import MedicalRAGPipeline


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical RAG API",
    description="Biomedical RAG pipeline — BioASQ Synergy 14",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance — loaded once at startup
pipeline: Optional[MedicalRAGPipeline] = None


# ---------------------------------------------------------------------------
# Startup: load pipeline
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global pipeline
    # Use fullpipeline.yaml as the default; override with env var RAG_CONFIG if needed
    import os
    config_path = Path(os.getenv("RAG_CONFIG", "configs/fullpipeline.yaml"))
    if not config_path.exists():
        print(f"[WARN] Config not found at {config_path} — pipeline not loaded")
        return
    with open(config_path) as f:
        config = yaml.safe_load(f)
    pipeline = MedicalRAGPipeline(config)
    print(f"[OK] Medical RAG Pipeline loaded from {config_path}")


# ---------------------------------------------------------------------------
# Helper: map pipeline output → SourceItem list
# ---------------------------------------------------------------------------

def _docs_to_sources(docs: list, max_sources: int = 10) -> List[SourceItem]:
    """Convert pipeline final_documents to the mobile SourceItem schema."""
    sources = []
    for doc in docs[:max_sources]:
        doc_id = doc.get("doc_id", "")
        score = float(doc.get("score") or doc.get("rerank_score") or 0.0)

        # Clamp score to [0, 1] — cross-encoder logits can exceed 1
        confidence = max(0.0, min(1.0, score))

        # Build a PubMed URL when the doc_id is numeric (PMID)
        url = None
        if doc_id and str(doc_id).isdigit():
            url = f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"

        # Use the first 300 chars of the abstract as the readable snippet
        abstract = doc.get("abstract") or ""
        page = (abstract[:300] + "…") if len(abstract) > 300 else abstract or None

        sources.append(SourceItem(
            title=doc.get("title") or f"PMID {doc_id}",
            url=url,
            page=page,
            confidence=round(confidence, 4),
        ))
    return sources


# ---------------------------------------------------------------------------
# Mobile API  — /api/v1
# ---------------------------------------------------------------------------

v1 = APIRouter(prefix="/api/v1")


@v1.get("/health")
async def v1_health():
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@v1.post("/query/submit", response_model=MobileQueryResponse)
async def v1_submit_query(
    request: MobileQueryRequest,
    authorization: Optional[str] = Header(None),   # Firebase JWT — accepted, not validated
):
    """
    Submit a biomedical question to the RAG pipeline.

    The Authorization header (Firebase JWT) is accepted and forwarded
    for logging purposes but not validated server-side.  All questions
    are processed regardless of token contents.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "SERVICE_UNAVAILABLE", "message": "Pipeline is not loaded"},
        )

    t0 = time.time()
    try:
        result = pipeline.process_query(
            query_text=request.question,
            top_k=request.topK,
            use_mmr=True,
            recency_boost=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": str(exc)},
        )

    latency_ms = int((time.time() - t0) * 1000)

    sources: List[SourceItem] = []
    if request.includeSources:
        docs = result.get("final_documents") or result.get("reranked_documents") or []
        sources = _docs_to_sources(docs, max_sources=request.topK)

    return MobileQueryResponse(
        id=str(uuid.uuid4()),
        question=request.question,
        answer=result.get("answer") or "",
        sources=sources,
        createdAt=datetime.utcnow().isoformat() + "Z",
        traceId=result.get("run_manifest_id"),
        latencyMs=latency_ms,
    )


@v1.get("/query/history", response_model=HistoryPage)
async def v1_get_history(
    page: int = 1,
    pageSize: int = 20,
    authorization: Optional[str] = Header(None),
):
    """
    Query history.  The mobile app caches history locally in Hive and only
    calls this endpoint to sync; returning an empty list is safe — the app
    will display its local cache.  Persistent server-side history can be
    added later (e.g. a SQLite/PostgreSQL store keyed by Firebase UID).
    """
    return HistoryPage(items=[], page=page, pageSize=pageSize, total=0)


@v1.delete("/query/history/{query_id}", status_code=204)
async def v1_delete_query(query_id: str, authorization: Optional[str] = Header(None)):
    """Delete a single history entry (no-op — history is client-side)."""
    return None


@v1.delete("/query/history", status_code=204)
async def v1_clear_history(authorization: Optional[str] = Header(None)):
    """Clear all history (no-op — history is client-side)."""
    return None


# Stub auth endpoints so the app does not get 404s on startup probes.
# The Flutter app uses Firebase Auth directly; these are never actually called.
@v1.post("/auth/login", status_code=200)
async def v1_auth_login():
    raise HTTPException(
        status_code=501,
        detail={"code": "NOT_IMPLEMENTED", "message": "Auth is handled by Firebase on the client."},
    )


@v1.post("/auth/register", status_code=201)
async def v1_auth_register():
    raise HTTPException(
        status_code=501,
        detail={"code": "NOT_IMPLEMENTED", "message": "Auth is handled by Firebase on the client."},
    )


@v1.get("/user/profile", status_code=200)
async def v1_get_profile(authorization: Optional[str] = Header(None)):
    raise HTTPException(
        status_code=501,
        detail={"code": "NOT_IMPLEMENTED", "message": "Profile is stored in Firestore on the client."},
    )


app.include_router(v1)


# ---------------------------------------------------------------------------
# Legacy internal endpoints  (unchanged)
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Medical RAG API", "version": "2.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
async def legacy_query(request: QueryRequest):
    """Legacy query endpoint — preserved for internal tooling."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = pipeline.process_query(
            query_text=request.query,
            top_k=request.top_k,
            use_mmr=request.use_mmr,
            recency_boost=request.recency_boost,
        )
        docs = result.get("final_documents") or result.get("reranked_documents") or []
        formatted = [
            Document(
                doc_id=str(d.get("doc_id", "")),
                title=d.get("title") or "",
                abstract=d.get("abstract") or "",
                pub_date=d.get("pub_date"),
                score=float(d.get("score") or 0.0),
                relevance_rank=i + 1,
            )
            for i, d in enumerate(docs)
        ]
        return QueryResponse(
            query=request.query,
            answer=result.get("answer") or "",
            retrieved_documents=formatted,
            run_manifest_id=result.get("run_manifest_id") or "",
            metadata=result.get("metadata") or {},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/config")
async def get_config():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
