"""
Hybrid retriever combining MedCPT dense (FAISS) and BM25 sparse retrieval
"""

from typing import List, Dict, Any
import numpy as np

from src.retrieval.medcpt_retriever import MedCPTRetriever
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever


class HybridMedCPTRetriever:
    """Hybrid retriever combining MedCPT dense retrieval and BM25 lexical retrieval"""

    def __init__(self, faiss_index: FAISSIndex, bm25_retriever: BM25Retriever, alpha: float = 0.5):
        """
        Args:
            faiss_index: FAISS index for dense retrieval
            bm25_retriever: BM25 retriever for sparse retrieval
            alpha: Weight for combining scores (1.0 = pure dense, 0.0 = pure sparse)
        """
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.alpha = float(alpha)
        self.medcpt_dense = MedCPTRetriever(faiss_index)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        top_k_final: int = 50,
        entities: List[Dict[str, Any]] | None = None,
        entity_boost: float = 2.0,
        max_entities: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using MedCPT dense + BM25 sparse hybrid
        """
        # Dense via FAISS using MedCPT embeddings
        dense_results = self.medcpt_dense.retrieve(query_embedding=query_embedding, top_k=top_k_dense)

        # Sparse via BM25
        sparse_results = self.bm25_retriever.search(
            query,
            top_k_sparse,
            entities=entities,
            entity_boost=entity_boost,
            max_entities=max_entities,
        )

        combined: Dict[str, Dict[str, Any]] = {}

        # Add dense scores
        for item in dense_results:
            doc_id = str(item["doc_id"])
            combined[doc_id] = {
                "dense_score": float(item.get("dense_score", item.get("score", 0.0))),
                "sparse_score": 0.0,
                "index": item.get("index")
            }

        # Add sparse scores
        for item in sparse_results:
            doc_id = str(item["doc_id"])
            if doc_id not in combined:
                combined[doc_id] = {
                    "dense_score": 0.0,
                    "sparse_score": float(item.get("score", 0.0)),
                    "index": None,
                    "source": item.get("source")
                }
            else:
                combined[doc_id]["sparse_score"] = float(item.get("score", 0.0))
                combined[doc_id]["source"] = item.get("source")

        # Normalize dense and sparse scores to 0-1
        dense_vals = [v["dense_score"] for v in combined.values()]
        max_dense = max(dense_vals) if dense_vals and max(dense_vals) > 0 else 1.0
        for did in combined:
            if max_dense > 0:
                combined[did]["dense_score"] = combined[did]["dense_score"] / max_dense
        
        sparse_vals = [v["sparse_score"] for v in combined.values()]
        max_sparse = max(sparse_vals) if sparse_vals and max(sparse_vals) > 0 else 1.0

        for did in combined:
            if max_sparse > 0:
                combined[did]["sparse_score"] = combined[did]["sparse_score"] / max_sparse

        # Combine
        for did in combined:
            ds = combined[did]["dense_score"]
            ss = combined[did]["sparse_score"]
            combined[did]["combined_score"] = self.alpha * ds + (1.0 - self.alpha) * ss

        # Sort and trim
        items = sorted(combined.items(), key=lambda x: x[1]["combined_score"], reverse=True)[:top_k_final]

        # Format results
        results: List[Dict[str, Any]] = []
        for did, s in items:
            results.append({
                "doc_id": did,
                "score": s["combined_score"],
                "dense_score": s["dense_score"],
                "sparse_score": s["sparse_score"],
                "index": s.get("index"),
                "source": s.get("source")
            })
        return results
