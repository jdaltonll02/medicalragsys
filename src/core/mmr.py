"""
Maximal Marginal Relevance (MMR) for diversity in retrieval
"""

import numpy as np
from typing import List, Tuple


def compute_mmr(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 10,
    recency_scores: np.ndarray = None,
    recency_weight: float = 0.0
) -> List[int]:
    """
    Compute Maximal Marginal Relevance to balance relevance and diversity
    
    Args:
        query_embedding: Query embedding vector (1D array)
        candidate_embeddings: Candidate document embeddings (2D array, shape: [n_docs, embedding_dim])
        lambda_param: Trade-off between relevance and diversity (0-1)
                     1.0 = pure relevance, 0.0 = pure diversity
        top_k: Number of documents to select
        recency_scores: Optional recency scores for temporal boosting (1D array)
        recency_weight: Weight for recency component (0-1)
    
    Returns:
        List of selected document indices in order
    """
    n_docs = len(candidate_embeddings)
    top_k = min(top_k, n_docs)
    
    # Normalize embeddings
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_embeddings = candidate_embeddings / (norms + 1e-8)
    
    # Compute relevance scores (cosine similarity with query)
    relevance_scores = np.dot(candidate_embeddings, query_embedding)
    
    # Add recency boost if provided
    if recency_scores is not None and recency_weight > 0:
        relevance_scores = (1 - recency_weight) * relevance_scores + recency_weight * recency_scores
    
    # Initialize selected and remaining indices
    selected_indices = []
    remaining_indices = list(range(n_docs))
    
    # Select first document (highest relevance)
    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select documents
    for _ in range(top_k - 1):
        if not remaining_indices:
            break
        
        mmr_scores = []
        
        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]
            
            # Diversity component (max similarity to already selected documents)
            similarities_to_selected = []
            for selected_idx in selected_indices:
                sim = np.dot(candidate_embeddings[idx], candidate_embeddings[selected_idx])
                similarities_to_selected.append(sim)
            
            max_sim_to_selected = max(similarities_to_selected) if similarities_to_selected else 0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((idx, mmr_score))
        
        # Select document with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices


def compute_recency_scores(pub_dates: List[str], decay_rate: float = 0.1) -> np.ndarray:
    """
    Compute recency scores based on publication dates
    More recent documents get higher scores
    
    Args:
        pub_dates: List of publication dates in YYYY-MM-DD format
        decay_rate: Exponential decay rate for older documents
    
    Returns:
        Recency scores (normalized to 0-1)
    """
    from datetime import datetime
    
    # Parse dates
    dates = []
    for date_str in pub_dates:
        try:
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except:
            # If parsing fails, use a very old date
            dates.append(datetime(1900, 1, 1))
    
    # Compute days since most recent document
    max_date = max(dates)
    days_old = np.array([(max_date - d).days for d in dates])
    
    # Apply exponential decay
    recency_scores = np.exp(-decay_rate * days_old / 365.0)
    
    # Normalize to 0-1 range
    recency_scores = (recency_scores - recency_scores.min()) / (recency_scores.max() - recency_scores.min() + 1e-8)
    
    return recency_scores
