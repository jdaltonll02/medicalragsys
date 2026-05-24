"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request schema for query endpoint"""
    query: str = Field(..., description="Medical question or query")
    top_k: int = Field(10, description="Number of documents to retrieve", ge=1, le=100)
    use_mmr: bool = Field(True, description="Whether to apply MMR for diversity")
    recency_boost: bool = Field(True, description="Whether to boost recent documents")


class Document(BaseModel):
    """Schema for a retrieved document"""
    doc_id: str
    title: str
    abstract: str
    pub_date: Optional[str] = None
    doi: Optional[str] = None
    score: float
    snippet: Optional[str] = None
    relevance_rank: int


class QueryResponse(BaseModel):
    """Response schema for query endpoint"""
    query: str
    answer: str
    retrieved_documents: List[Document]
    run_manifest_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    pipeline_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
