"""
MedCPT encoder for medical documents and queries
"""

import numpy as np
from typing import List, Union


import logging as _logging
_encoder_logger = _logging.getLogger(__name__)


class MedCPTEncoder:
    """MedCPT encoder for medical domain embeddings.

    Uses asymmetric encoding: ncbi/MedCPT-Query-Encoder for queries and
    ncbi/MedCPT-Article-Encoder for documents, matching MedCPT's training setup.
    """

    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Query-Encoder",
        article_model_name: str = "ncbi/MedCPT-Article-Encoder",
        device: str = "auto",
    ):
        """
        Initialize MedCPT encoder

        Args:
            model_name: Query encoder HuggingFace model name
            article_model_name: Article encoder HuggingFace model name
            device: Device for inference (cpu/cuda/auto)
        """
        self.model_name = model_name
        self.article_model_name = article_model_name

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            # Honour the requested device, but verify CUDA is actually usable
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        _encoder_logger.warning(
                            "device='cuda' requested but CUDA is unavailable; falling back to CPU"
                        )
                        device = "cpu"
                except Exception:
                    device = "cpu"
            self.device = device

        _encoder_logger.info(f"MedCPT encoder using device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.article_model = None
        self.article_tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load query and article MedCPT models."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            # Benchmark mode: let cuDNN pick fastest convolution algorithm for fixed input sizes
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            if self.device == "cuda":
                self.model = self.model.half()  # fp16 halves memory, ~2× throughput
            self.model.to(self.device)
            self.model.eval()
            _encoder_logger.info(f"Loaded query encoder: {self.model_name}")
        except Exception as e:
            _encoder_logger.warning(f"Could not load query encoder '{self.model_name}': {e}")
            self.model = None

        # Load article encoder; fall back to query encoder if unavailable
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.article_tokenizer = AutoTokenizer.from_pretrained(self.article_model_name)
            self.article_model = AutoModel.from_pretrained(self.article_model_name)
            if self.device == "cuda":
                self.article_model = self.article_model.half()
            self.article_model.to(self.device)
            self.article_model.eval()
            _encoder_logger.info(f"Loaded article encoder: {self.article_model_name}")
        except Exception as e:
            _encoder_logger.warning(
                f"Could not load article encoder '{self.article_model_name}': {e}. "
                "Falling back to query encoder for documents."
            )
            self.article_model = None
            self.article_tokenizer = None
    
    def encode(
        self,
        texts: Union[str, List[str], List[List[str]]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode document texts using the article encoder.

        Args:
            texts: Single text, list of texts, or list of [title, abstract] pairs.
                   MedCPT-Article-Encoder expects [title, abstract] pairs for best results:
                   the tokenizer encodes them as [CLS] title [SEP] abstract [SEP].
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use article encoder if available, otherwise fall back to query encoder
        model = self.article_model if self.article_model is not None else self.model
        tokenizer = self.article_tokenizer if self.article_tokenizer is not None else self.tokenizer
        is_article = self.article_model is not None

        if model is None:
            embeddings = np.random.randn(len(texts), 768).astype(np.float32)
        else:
            embeddings = self._encode_batch(texts, batch_size, model=model, tokenizer=tokenizer,
                                            use_pair_encoding=is_article)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def _encode_batch(self, texts, batch_size: int, model=None, tokenizer=None,
                      use_pair_encoding: bool = False) -> np.ndarray:
        """Encode texts in batches using the given model and tokenizer.

        Args:
            texts: List of strings OR list of [title, abstract] pairs.
            use_pair_encoding: If True and texts contains pairs, pass them as sentence pairs
                               to the tokenizer so [CLS] title [SEP] abstract [SEP] is used —
                               matching MedCPT-Article-Encoder's training format.
        """
        import torch

        if model is None:
            model = self.model
        if tokenizer is None:
            tokenizer = self.tokenizer

        use_amp = self.device == "cuda"
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_items = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if batch_num % 50 == 0 or batch_num == num_batches:
                _encoder_logger.info(
                    f"Encoding batch {batch_num}/{num_batches} "
                    f"({min(i + batch_size, len(texts))}/{len(texts)} texts)"
                )

            # Detect if this batch contains (title, abstract) pairs
            has_pairs = use_pair_encoding and batch_items and isinstance(batch_items[0], (list, tuple))
            if has_pairs:
                # Sentence-pair tokenization: [CLS] title [SEP] abstract [SEP]
                titles = [item[0] if len(item) > 0 else "" for item in batch_items]
                abstracts = [item[1] if len(item) > 1 else "" for item in batch_items]
                inputs = tokenizer(
                    titles,
                    abstracts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Plain text tokenization (queries or fallback)
                batch_texts = [item if isinstance(item, str) else " ".join(item) for item in batch_items]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                # Cast to float32 before moving to CPU — avoids numpy fp16 issues
                embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query using the query encoder.

        Args:
            query: Query text
            normalize: Whether to L2-normalize embedding

        Returns:
            Query embedding (1D array)
        """
        if self.model is None:
            return np.random.randn(768).astype(np.float32)

        embeddings = self._encode_batch([query], batch_size=1, model=self.model, tokenizer=self.tokenizer)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings[0]
