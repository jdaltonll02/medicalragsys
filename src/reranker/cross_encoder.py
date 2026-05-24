"""
Cross-encoder reranker for retrieved documents
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class CrossEncoderReranker:
    """Cross-encoder reranker using S-PubMedBERT-MS-MARCO"""
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO", batch_size: int = 16, device: str = "auto"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            # Resolve device: 'auto' -> cuda if available else cpu
            dev = self.device
            if dev == "auto":
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
            elif dev == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        dev = "cpu"
                except Exception:
                    dev = "cpu"
            self.model = CrossEncoder(self.model_name, device=dev)
            # Enforce safe max sequence length to prevent positional embedding overflow
            try:
                tok_max = getattr(self.model.tokenizer, "model_max_length", None)
                if tok_max is None or tok_max > 100000 or tok_max == float("inf"):
                    tok_max = 512
                # sentence-transformers honors this attribute for internal tokenization
                self.model.max_seq_length = int(tok_max)
                self._max_seq_length = int(tok_max)
            except Exception:
                self._max_seq_length = 512
        except Exception as e:
            print(f"Warning: Could not load cross-encoder model: {e}")
            print("Using placeholder reranker")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_field: str = "abstract",
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query text
            documents: List of documents with text field
            text_field: Field name containing document text
            top_k: Number of documents to return
        
        Returns:
            Reranked list of documents with updated scores
        """
        if self.model is None or not documents:
            # Return original documents if model not loaded
            return documents[:top_k]
        
        # Prepare query-document pairs with safe truncation to avoid >512 token sequences
        pairs = []
        texts = []
        for doc in documents:
            txt = doc.get(text_field, "") or ""
            texts.append(txt)
            pairs.append((query, txt))

        # Truncate long texts deterministically with tokenizer to ensure <= max_len tokens
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_name)
            max_len = getattr(tok, "model_max_length", 512) or 512
            if max_len > 100000 or max_len == float("inf"):
                max_len = 512
            reserve = 32
            q_ids = tok.encode(query, add_special_tokens=False)
            q_allow = max(8, min(len(q_ids), (max_len - reserve) // 3))
            q_ids = q_ids[:q_allow]
            q_text = tok.decode(q_ids, skip_special_tokens=True)
            p_allow = max_len - reserve - len(q_ids)
            def trunc_passage(p: str) -> str:
                p_ids = tok.encode(p, add_special_tokens=False)
                p_ids = p_ids[: max(8, p_allow)]
                return tok.decode(p_ids, skip_special_tokens=True)
            pairs = [(q_text, trunc_passage(t)) for t in texts]
        except Exception:
            # Fallback: naive char truncation to ~2048 chars
            pairs = [(query, (t[:2048] if isinstance(t, str) else "")) for t in texts]
        
        # Get reranking scores
        # Pass max_length to sentence-transformers predict to enforce truncation
        # sentence-transformers v2.x uses self.model.max_seq_length; passing max_length is unsupported
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Add scores to documents
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0.0)
            doc_copy["score"] = float(score)  # Update score
            reranked_docs.append(doc_copy)
        
        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_docs[:top_k]
    
    def score_pairs(self, query: str, texts: List[str]) -> np.ndarray:
        """
        Score query-text pairs
        
        Args:
            query: Query text
            texts: List of texts to score
        
        Returns:
            Array of scores
        """
        if self.model is None:
            # Return dummy scores
            return np.random.rand(len(texts))
        
        # Prepare pairs with truncation safeguards
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_name)
            max_len = getattr(tok, "model_max_length", 512) or 512
            if max_len > 100000 or max_len == float("inf"):
                max_len = 512
            def truncate(s: str, limit: int) -> str:
                return s[: max(1, limit * 4)]
            pairs = [(query, truncate(text, max_len)) for text in texts]
        except Exception:
            pairs = [(query, (text[:2048] if isinstance(text, str) else "")) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        return np.array(scores)
