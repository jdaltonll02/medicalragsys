"""
FastAPI application for Medical RAG System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import yaml
from pathlib import Path

from src.api.schemas import QueryRequest, QueryResponse, Document
from src.pipeline.med_rag import MedicalRAGPipeline


app = FastAPI(
    title="Medical RAG API",
    description="API for Medical Retrieval-Augmented Generation System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[MedicalRAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global pipeline
    
    config_path = Path("configs/pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    pipeline = MedicalRAGPipeline(config)
    print("Medical RAG Pipeline initialized")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a medical query through the RAG pipeline
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.process_query(
            query_text=request.query,
            top_k=request.top_k,
            use_mmr=request.use_mmr,
            recency_boost=request.recency_boost
        )
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            retrieved_documents=result["retrieved_documents"],
            run_manifest_id=result["run_manifest_id"],
            metadata=result.get("metadata", {})
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current pipeline configuration"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
