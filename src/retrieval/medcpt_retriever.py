"""
MedCPT dense retriever using FAISS
"""

import numpy as np
from typing import List, Dict, Any

from src.retrieval.faiss_index import FAISSIndex


class MedCPTRetriever:
    """Dense retriever that queries FAISS using MedCPT embeddings"""

    def __init__(self, faiss_index: FAISSIndex):
        self.faiss_index = faiss_index

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve top documents using FAISS on MedCPT embeddings.

        Args:
            query_embedding: 1D numpy array for the query
            top_k: number of results to return

        Returns:
            List of dicts: {doc_id, score, dense_score, index}
        """
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results: List[Dict[str, Any]] = []
        if indices is None or len(indices) == 0:
            return results
        for score, idx in zip(scores, indices):
            try:
                if hasattr(self.faiss_index, "doc_ids") and self.faiss_index.doc_ids and int(idx) < len(self.faiss_index.doc_ids):
                    doc_id = str(self.faiss_index.doc_ids[int(idx)])
                else:
                    doc_id = str(int(idx))
            except Exception:
                doc_id = str(int(idx))
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "dense_score": float(score),
                "index": int(idx),
            })
        return results
