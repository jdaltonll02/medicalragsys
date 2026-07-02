"""
Hybrid retriever combining FAISS and BM25
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval"""
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            faiss_index: FAISS index for dense retrieval
            bm25_retriever: BM25 retriever for sparse retrieval
            alpha: Weight for combining scores (0-1, 1.0 = pure dense, 0.0 = pure sparse)
        """
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
    
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        top_k_final: int = 50,
        entities: list | None = None,
        entity_boost: float = 2.0,
        max_entities: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            top_k_dense: Number of results from dense retrieval
            top_k_sparse: Number of results from sparse retrieval
            top_k_final: Number of final results
        
        Returns:
            List of retrieved documents with combined scores
        """
        # Dense retrieval (FAISS)
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k_dense)
        
        # Sparse retrieval (BM25)
        if self.bm25_retriever is not None:
            sparse_results = self.bm25_retriever.search(
                query,
                top_k_sparse,
                entities=entities,
                entity_boost=entity_boost,
                max_entities=max_entities,
            )
        else:
            sparse_results = []
        
        # Combine results
        combined_scores = {}
        
        # Add dense scores
        for idx, score in zip(dense_indices, dense_scores):
            idx = int(idx)
            if idx < 0:
                # FAISS -1 sentinel — should be filtered in FAISSIndex.search(), but
                # guard here as well so callers never store a garbage PMID.
                continue
            # Prefer external doc_id mapping from FAISS if available
            doc_ids = getattr(self.faiss_index, "doc_ids", None)
            if doc_ids and idx < len(doc_ids):
                doc_id = str(doc_ids[idx])
            else:
                doc_id = str(idx)
            combined_scores[doc_id] = {
                "dense_score": float(score),
                "sparse_score": 0.0,
                "index": idx
            }
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result["doc_id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                    "index": None,
                    "source": result.get("source")
                }
            else:
                combined_scores[doc_id]["sparse_score"] = result["score"]
                if combined_scores[doc_id].get("source") is None:
                    combined_scores[doc_id]["source"] = result.get("source")
        
        # Normalize scores to [0, 1] so they can be linearly combined.
        #
        # Dense (FAISS/cosine): scores are in [-1, 1].
        #   - When BM25 is active: use min-max normalization so both retrievers
        #     contribute on the same scale.
        #   - When BM25 is absent (FAISS-only, alpha=1.0): skip normalization and
        #     shift cosine scores to [0, 1] by adding 1 and halving. Min-max would
        #     collapse to 0 whenever all top-200 scores cluster within 1e-9 of each
        #     other, making the ranking arbitrary and the combined score 0.
        #
        # Sparse (BM25): scores are always ≥ 0, so max-normalization is safe.

        faiss_only = self.bm25_retriever is None

        faiss_scores = [v["dense_score"] for v in combined_scores.values()
                        if v.get("index") is not None]
        if faiss_scores:
            if faiss_only:
                # Shift cosine similarities from [-1,1] → [0,1]; preserves ranking.
                for v in combined_scores.values():
                    if v.get("index") is not None:
                        v["dense_score"] = (v["dense_score"] + 1.0) / 2.0
            else:
                # Min-max normalization so FAISS and BM25 land on the same scale.
                lo = min(faiss_scores)
                hi = max(faiss_scores)
                rng = hi - lo
                for v in combined_scores.values():
                    if v.get("index") is not None:
                        v["dense_score"] = (v["dense_score"] - lo) / rng if rng > 1e-9 else 0.5
                    else:
                        # BM25-only doc — not retrieved by FAISS, sentinel stays 0
                        v["dense_score"] = 0.0

        # Sparse: max-normalization (scores are always ≥ 0)
        sparse_values = [v["sparse_score"] for v in combined_scores.values()]
        if sparse_values:
            max_sparse = max(sparse_values)
            if max_sparse > 1e-9:
                for v in combined_scores.values():
                    v["sparse_score"] /= max_sparse
        
        # Compute combined scores
        for doc_id in combined_scores:
            dense = combined_scores[doc_id]["dense_score"]
            sparse = combined_scores[doc_id]["sparse_score"]
            combined_scores[doc_id]["combined_score"] = self.alpha * dense + (1 - self.alpha) * sparse
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )[:top_k_final]

        # Format results
        results = []
        for doc_id, scores in sorted_results:
            results.append({
                "doc_id": doc_id,
                "score": scores["combined_score"],
                "dense_score": scores["dense_score"],
                "sparse_score": scores["sparse_score"],
                "index": scores["index"],
                "source": scores.get("source")
            })
        
        return results
