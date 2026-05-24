"""
Integration smoke tests for the RAG pipeline
"""

import pytest
import yaml
from pathlib import Path
import os

from src.pipeline.med_rag import MedicalRAGPipeline


@pytest.fixture
def config():
    """Load test configuration"""
    config_path = Path("configs/pipeline_config.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Override for testing
    cfg["llm"]["provider"] = "stub"
    os.environ["LLM_PROVIDER"] = "stub"
    
    return cfg


@pytest.fixture
def pipeline(config):
    """Initialize pipeline for testing"""
    return MedicalRAGPipeline(config)


def test_pipeline_initialization(pipeline):
    """Test that pipeline initializes correctly"""
    assert pipeline is not None
    assert pipeline.config is not None
    assert pipeline.encoder is not None
    assert pipeline.llm is not None


def test_process_simple_query(pipeline):
    """Test processing a simple query"""
    query = "What are the symptoms of COVID-19?"
    
    result = pipeline.process_query(query, top_k=5)
    
    assert "query" in result
    assert "answer" in result
    assert "retrieved_documents" in result
    assert "run_manifest_id" in result
    assert result["query"] == query
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_query_with_entities(pipeline):
    """Test query processing extracts entities"""
    query = "What are the treatments for diabetes mellitus?"
    
    result = pipeline.process_query(query)
    
    assert "entities" in result
    assert isinstance(result["entities"], list)


def test_query_normalization(pipeline):
    """Test query normalization"""
    query = "  What  are   the   symptoms?  "
    
    result = pipeline.process_query(query)
    
    assert "normalized_query" in result
    # Should remove extra whitespace
    assert "   " not in result["normalized_query"]


def test_empty_query(pipeline):
    """Test handling of empty query"""
    query = ""
    
    result = pipeline.process_query(query)
    
    # Should still return a valid response structure
    assert "answer" in result
    assert "retrieved_documents" in result


def test_metadata_tracking(pipeline):
    """Test that metadata is tracked correctly"""
    query = "What is hypertension?"
    
    result = pipeline.process_query(query)
    
    assert "metadata" in result
    assert "num_retrieved" in result["metadata"]
    assert "num_final" in result["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
