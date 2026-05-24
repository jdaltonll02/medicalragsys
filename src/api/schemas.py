"""
Pydantic schemas for API requests and responses.

Two schema families live here:
  1. Mobile contract  — /api/v1/query/submit  (used by the Flutter app)
  2. Legacy internal  — /query                (kept for backward compatibility)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Mobile contract schemas  (matches BACKEND_CONTRACT_SPEC.yaml)
# ---------------------------------------------------------------------------

class MobileQueryRequest(BaseModel):
    """Request body for POST /api/v1/query/submit"""
    question: str = Field(..., description="Biomedical question from the user")
    sessionId: Optional[str] = Field(None, max_length=200, description="Optional client session ID")
    topK: int = Field(5, ge=1, le=20, description="Number of source documents to return")
    includeSources: bool = Field(True, description="Whether to include source documents")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional retrieval filters (reserved)")


class SourceItem(BaseModel):
    """A single retrieved source document as returned to the mobile app"""
    title: str
    url: Optional[str] = None       # PubMed URL constructed from PMID
    page: Optional[str] = None      # Abstract excerpt used as snippet
    confidence: float               # Normalised relevance score  0.0–1.0


class MobileQueryResponse(BaseModel):
    """Response body for POST /api/v1/query/submit"""
    id: str                         # UUID for this query result
    question: str
    answer: str
    sources: List[SourceItem]
    createdAt: str                  # ISO-8601 datetime string
    traceId: Optional[str] = None   # Maps to run_manifest_id from pipeline
    latencyMs: Optional[int] = None


class HistoryPage(BaseModel):
    """Response body for GET /api/v1/query/history"""
    items: List[MobileQueryResponse]
    page: int
    pageSize: int
    total: int


class ErrorResponse(BaseModel):
    """Standard error shape expected by the mobile app"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    requestId: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# Legacy internal schemas  (POST /query — kept for tooling / backward compat)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Legacy request schema"""
    query: str = Field(..., description="Medical question or query")
    top_k: int = Field(10, ge=1, le=100)
    use_mmr: bool = Field(True)
    recency_boost: bool = Field(True)


class Document(BaseModel):
    """Legacy retrieved-document schema"""
    doc_id: str
    title: str
    abstract: str
    pub_date: Optional[str] = None
    doi: Optional[str] = None
    score: float
    snippet: Optional[str] = None
    relevance_rank: int


class QueryResponse(BaseModel):
    """Legacy response schema"""
    query: str
    answer: str
    retrieved_documents: List[Document]
    run_manifest_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
