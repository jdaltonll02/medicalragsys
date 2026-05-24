"""
FAISS index for dense vector retrieval
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
from pathlib import Path


class FAISSIndex:
    """FAISS index wrapper for dense retrieval"""
    
    def __init__(self, index_path: Optional[str] = None, embedding_dim: int = 768):
        """
        Initialize FAISS index
        
        Args:
            index_path: Path to existing FAISS index file
            embedding_dim: Dimension of embeddings
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = None
        # Maintain an optional mapping from FAISS row index -> external doc_id
        self.doc_ids: List[str] = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        else:
            self._create_index()
    
    def _create_index(self):
        """Create a new FAISS index"""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def load_index(self, index_path: str):
        """Load FAISS index from file"""
        self.index = faiss.read_index(index_path)
        self.embedding_dim = self.index.d
    
    def save_index(self, output_path: str):
        """Save FAISS index to file"""
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, output_path)
    
    def add_vectors(self, vectors: np.ndarray):
        """
        Add vectors to the index
        
        Args:
            vectors: Numpy array of shape (n_vectors, embedding_dim)
        """
        if self.index is None:
            self._create_index()
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / (norms + 1e-8)
        
        self.index.add(normalized_vectors.astype(np.float32))
        # Ensure doc_ids length stays in sync when caller sets mapping
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector (1D array)
            top_k: Number of results to return
        
        Returns:
            Tuple of (scores, indices)
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1)
        norm = np.linalg.norm(query_vector)
        query_vector = query_vector / (norm + 1e-8)
        
        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        return scores[0], indices[0]

    def set_doc_ids(self, doc_ids: List[str]):
        """Set external document IDs aligned to FAISS rows.
        Caller must ensure the order matches add_vectors insertion order.
        """
        self.doc_ids = list(doc_ids or [])
    
    def get_num_vectors(self) -> int:
        """Get number of vectors in the index"""
        return self.index.ntotal if self.index is not None else 0
