"""
Unit tests for MMR implementation
"""

import pytest
import numpy as np
from src.core.mmr import compute_mmr, compute_recency_scores


def test_compute_mmr_basic():
    """Test basic MMR computation"""
    # Create simple embeddings
    query_embedding = np.array([1.0, 0.0])
    candidate_embeddings = np.array([
        [1.0, 0.0],  # Very similar to query
        [0.9, 0.1],  # Similar to query
        [0.0, 1.0],  # Orthogonal to query
    ])
    
    # Compute MMR
    selected = compute_mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        lambda_param=0.7,
        top_k=2
    )
    
    assert len(selected) == 2
    assert 0 in selected  # Most relevant should be selected
    assert isinstance(selected, list)


def test_compute_mmr_with_recency():
    """Test MMR with recency scores"""
    query_embedding = np.array([1.0, 0.0])
    candidate_embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
    ])
    
    recency_scores = np.array([0.5, 0.8, 1.0])  # Last doc is most recent
    
    selected = compute_mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        lambda_param=0.5,
        top_k=2,
        recency_scores=recency_scores,
        recency_weight=0.3
    )
    
    assert len(selected) == 2


def test_compute_recency_scores():
    """Test recency score computation"""
    pub_dates = ["2020-01-01", "2022-01-01", "2023-01-01"]
    
    scores = compute_recency_scores(pub_dates)
    
    assert len(scores) == 3
    assert scores[-1] >= scores[0]  # Most recent should have higher score
    assert 0.0 <= scores.min() <= 1.0
    assert 0.0 <= scores.max() <= 1.0


def test_mmr_empty_candidates():
    """Test MMR with empty candidates"""
    query_embedding = np.array([1.0, 0.0])
    candidate_embeddings = np.array([]).reshape(0, 2)
    
    selected = compute_mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        top_k=5
    )
    
    assert len(selected) == 0


def test_mmr_top_k_exceeds_candidates():
    """Test MMR when top_k exceeds number of candidates"""
    query_embedding = np.array([1.0, 0.0])
    candidate_embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
    ])
    
    selected = compute_mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        top_k=10
    )
    
    assert len(selected) == 2  # Should return all candidates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
