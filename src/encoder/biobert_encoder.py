"""
BioBERT encoder for medical documents and queries
"""

import numpy as np
from typing import List, Union


class BioBERTEncoder:
    """BioBERT encoder for biomedical domain embeddings"""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", device: str = "auto"):
        """
        Initialize BioBERT encoder

        Args:
            model_name: HuggingFace model name (BioBERT or PubMedBERT)
            device: Device for inference (cpu/cuda)
        """
        self.model_name = model_name
        # Resolve device: 'auto' -> cuda if available else cpu
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load BioBERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load BioBERT model: {e}")
            print("Using placeholder encoder")
            self.model = None

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model is None:
            # Placeholder: return random embeddings
            embedding_dim = 768
            embeddings = np.random.randn(len(texts), embedding_dim).astype(np.float32)
        else:
            embeddings = self._encode_batch(texts, batch_size)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode texts in batches using CLS token"""
        import torch

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query

        Args:
            query: Query text
            normalize: Whether to normalize embedding

        Returns:
            Query embedding (1D array)
        """
        embeddings = self.encode([query], normalize=normalize)
        return embeddings[0]
