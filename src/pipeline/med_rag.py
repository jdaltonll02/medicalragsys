"""
Medical RAG Pipeline - Main orchestration
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from src.core.normalizer import normalize_medical_query
from src.core.mmr import compute_mmr, compute_recency_scores
from src.core.utils import set_random_seed, save_run_manifest
from src.ner.ner_service import NERService
from src.encoder.medcpt_encoder import MedCPTEncoder
from src.encoder.biobert_encoder import BioBERTEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.openai_client import OpenAIClient
from src.llm.stub_llm import StubLLM


class MedicalRAGPipeline:
    """End-to-end Medical RAG Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        
        # Set random seed for reproducibility
        seed = config.get("pipeline", {}).get("seed", 42)
        set_random_seed(seed)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # NER
        ner_config = self.config.get("ner", {})
        self.ner = NERService(
            model_name=ner_config.get("model", "en_core_sci_sm"),
            confidence_threshold=ner_config.get("confidence_threshold", 0.7)
        )
        
        # Encoder
        encoder_config = self.config.get("encoder", {})
        backend = encoder_config.get("backend", "medcpt").lower()
        if backend == "biobert":
            self.encoder = BioBERTEncoder(
                model_name=encoder_config.get("model", "dmis-lab/biobert-base-cased-v1.1"),
                device=encoder_config.get("device", "auto")
            )
        else:
            self.encoder = MedCPTEncoder(
                model_name=encoder_config.get("model", "ncbi/MedCPT-Query-Encoder"),
                article_model_name=encoder_config.get("article_model", "ncbi/MedCPT-Article-Encoder"),
                device=encoder_config.get("device", "auto")
            )
        
        # FAISS Index
        faiss_config = self.config.get("faiss", {})
        self.faiss_index = FAISSIndex(
            index_path=faiss_config.get("save_path"),
            embedding_dim=encoder_config.get("embedding_dim", 768)
        )
        
        # BM25 Retriever — skip entirely when Elasticsearch is disabled
        _fallback = self.config.get("fallback", {})
        _use_es = not _fallback.get("use_faiss_only", False) and _fallback.get("use_elasticsearch", True)
        if _use_es:
            bm25_config = self.config.get("bm25", {})
            self.bm25_retriever = BM25Retriever(
                host=bm25_config.get("elasticsearch_host", "localhost"),
                port=bm25_config.get("elasticsearch_port", 9200),
                index_name=bm25_config.get("index_name", "medical_docs_2")
            )
        else:
            logging.getLogger(__name__).info(
                "Elasticsearch disabled (use_faiss_only=true); skipping BM25Retriever init"
            )
            self.bm25_retriever = None

        # Hybrid Retriever
        retrieval_config = self.config.get("retrieval", {})
        self.hybrid_retriever = HybridRetriever(
            faiss_index=self.faiss_index,
            bm25_retriever=self.bm25_retriever,
            alpha=retrieval_config.get("alpha", 0.5)
        )
        
        # Reranker
        reranker_config = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            batch_size=reranker_config.get("batch_size", 16),
            device=reranker_config.get("device", "auto")
        )
        
        # LLM
        from src.llm.gemini_client import GeminiClient
        
        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "openai")
        
        if llm_provider == "stub" or os.getenv("LLM_PROVIDER") == "stub":
            self.llm = StubLLM()
        elif llm_provider == "gemini":
            self.llm = GeminiClient(
                model=llm_config.get("model", "gemini-2.0-flash"),
                api_key=llm_config.get("api_key"),
                project_id=llm_config.get("project_id"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024)
            )
        else:  # openai (default)
            self.llm = OpenAIClient(
                model=llm_config.get("model", "gpt-4"),
                api_key=llm_config.get("api_key"),
                base_url=llm_config.get("base_url"),
                project_id=llm_config.get("project_id"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024),
                prompt_for_key=llm_config.get("prompt_for_key", True),
                use_keyring=llm_config.get("use_keyring", True),
                save_to_keyring=llm_config.get("save_to_keyring", False)
            )


    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        reset_index: bool = True,
        index_fallback: bool = True
    ):
        """Build indices for dense (FAISS) and sparse (BM25) retrieval"""
        logger = logging.getLogger(__name__)
        if not documents:
            return
        # Keep an in-memory store to enrich retrieval results later
        if reset_index or not hasattr(self, "_doc_store"):
            self._doc_store = []
        self._doc_store.extend(documents)
        # Encode abstracts for FAISS using article encoder
        abstracts = [doc.get("abstract", "") for doc in documents]
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(abstracts)
        logger.info(f"Adding {len(embeddings)} vectors to FAISS...")
        self.faiss_index.add_vectors(embeddings)
        # Provide FAISS with external doc_id mapping aligned to insertion order
        try:
            doc_ids = [doc.get("doc_id") for doc in documents]
            if reset_index or not getattr(self.faiss_index, "doc_ids", None):
                self.faiss_index.set_doc_ids(doc_ids)
            else:
                self.faiss_index.doc_ids.extend(doc_ids)
        except Exception:
            pass
        # Index documents into Elasticsearch
        try:
            fallback_config = self.config.get("fallback", {})
            use_elasticsearch = not fallback_config.get("use_faiss_only", False) and fallback_config.get("use_elasticsearch", True)
            if not use_elasticsearch:
                logger.info("Skipping BM25/Elasticsearch indexing (fallback.use_faiss_only=true or use_elasticsearch=false)")
            else:
                if reset_index:
                    try:
                        self.bm25_retriever.reset_index()
                    except Exception:
                        pass
                logger.info(f"Indexing {len(documents)} documents into Elasticsearch...")
                self.bm25_retriever.index_documents(documents, index_fallback=index_fallback)
                logger.info("BM25 indexing complete.")
        except Exception as e:
            logger.warning(f"BM25 indexing failed: {e}")
    
    def process_query(
        self,
        query_text: str,
        top_k: int = 10,
        use_mmr: bool = True,
        recency_boost: bool = True,
        question_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query_text: User query
            top_k: Number of final documents
            use_mmr: Whether to apply MMR
            recency_boost: Whether to boost recent documents
            question_type: Type of question (yesno, factoid, list, summary) for better answer formatting
        
        Returns:
            Dictionary with answer and retrieved documents
        """
        run_manifest_id = str(uuid.uuid4())
        
        # 1. Normalize query
        norm_cfg = self.config.get("query_normalization", {})
        if not norm_cfg.get("enabled", True):
            # Normalizer fully disabled — use raw query (unicode-safe strip only)
            normalized_query = query_text.strip()
        else:
            normalized_query = normalize_medical_query(
                query_text,
                lowercase=norm_cfg.get("lowercase", True),
                remove_punct=norm_cfg.get("remove_punctuation", True),
            )
        
        # 2. Extract entities (NER)
        entities = self.ner.extract_entities(normalized_query)
        
        # 3. Encode query (optionally append top entities for dense retrieval signal)
        retrieval_config = self.config.get("retrieval", {})
        faiss_append = retrieval_config.get("faiss_entity_append", True)
        faiss_max_entities = retrieval_config.get("faiss_max_entities", 3)
        ent_texts = [e.get("text", "").strip() for e in entities if isinstance(e, dict)]
        ent_texts = [t for t in ent_texts if t]
        if faiss_append and ent_texts:
            augmented_query = (normalized_query + " " + " ".join(ent_texts[:faiss_max_entities])).strip()
        else:
            augmented_query = normalized_query
        query_embedding = self.encoder.encode_query(augmented_query)
        
        # 4. Hybrid retrieval
        retrieved_docs = self.hybrid_retriever.retrieve(
            query=normalized_query,
            query_embedding=query_embedding,
            top_k_dense=retrieval_config.get("top_k_dense", 100),
            top_k_sparse=retrieval_config.get("top_k_sparse", 100),
            top_k_final=retrieval_config.get("top_k_final", 50),
            entities=entities,
            entity_boost=retrieval_config.get("bm25_entity_boost", 2.0),
            max_entities=retrieval_config.get("bm25_max_entities", 5),
        )

        # Enrich retrieved docs with title/abstract using stored corpus or ES source
        enriched = []
        for item in retrieved_docs:
            doc = {
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "index": item.get("index")
            }
            # Prefer ES source if available
            src = item.get("source")
            if src and isinstance(src, dict):
                doc["title"] = src.get("title", "")
                doc["abstract"] = src.get("abstract", "")
                doc["pub_date"] = src.get("pub_date")
            else:
                idx = doc.get("index")
                if isinstance(idx, int) and hasattr(self, "_doc_store") and 0 <= idx < len(self._doc_store):
                    store_doc = self._doc_store[idx]
                    doc["title"] = store_doc.get("title", "")
                    doc["abstract"] = store_doc.get("abstract", "")
                    doc["pub_date"] = store_doc.get("pub_date")
            enriched.append(doc)
        retrieved_docs = enriched
        # Preserve the full pre-rerank set for evaluation metrics (Recall@K, etc.)
        retrieved_pre_rerank = list(retrieved_docs)
        
        # 5. Rerank
        reranker_config = self.config.get("reranker", {})
        reranked_docs = self.reranker.rerank(
            query=normalized_query,
            documents=retrieved_docs,
            top_k=reranker_config.get("top_k", 20)
        )
        
        # 6. Apply MMR for diversity
        if use_mmr and len(reranked_docs) > 0:
            mmr_config = self.config.get("mmr", {})
            temporal_config = self.config.get("temporal", {})
            effective_recency_boost = recency_boost and temporal_config.get("enabled", True)
            decay_rate = temporal_config.get("recency_decay", 0.1)
            doc_embeddings = self.encoder.encode(
                [doc.get("abstract", "") for doc in reranked_docs]
            )

            # Compute recency scores if needed
            recency_scores = None
            if effective_recency_boost:
                pub_dates = [doc.get("pub_date", "2000-01-01") for doc in reranked_docs]
                recency_scores = compute_recency_scores(pub_dates, decay_rate=decay_rate)
            
            selected_indices = compute_mmr(
                query_embedding=query_embedding,
                candidate_embeddings=doc_embeddings,
                lambda_param=mmr_config.get("lambda_param", 0.7),
                top_k=top_k,
                recency_scores=recency_scores,
                recency_weight=mmr_config.get("recency_weight", 0.3) if effective_recency_boost else 0.0
            )
            
            final_docs = [reranked_docs[i] for i in selected_indices]
        else:
            final_docs = reranked_docs[:top_k]
        
        # 7. Generate answer with LLM
        llm_config = self.config.get("llm", {})
        system_prompt = llm_config.get("system_prompt", "You are a medical AI assistant.")

        # Enrich final_docs with title/abstract before LLM call (needed in skip-indexing mode
        # where FAISS returns row indices but corpus text is fetched lazily via _corpus_lookup)
        corpus_lookup = getattr(self, "_corpus_lookup", None)
        if corpus_lookup is not None:
            for doc in final_docs:
                if not doc.get("abstract"):
                    row = doc.get("index")
                    if isinstance(row, int):
                        doc["title"], doc["abstract"] = corpus_lookup.get(row)

        # Use provided question_type or try to detect from query
        if question_type is None:
            query_lower = query_text.lower()
            if any(query_lower.startswith(word) for word in ["is ", "are ", "does ", "do ", "can ", "could ", "will ", "would "]):
                question_type = "yesno"
            elif "how many" in query_lower or "what is the" in query_lower or "who is" in query_lower or "which virus" in query_lower:
                question_type = "factoid"
            elif "list" in query_lower or "which are" in query_lower or "what are the" in query_lower:
                question_type = "list"
            elif "what is" in query_lower or "describe" in query_lower or "explain" in query_lower:
                question_type = "summary"
        
        answer = self.llm.generate_with_context(
            query=query_text,
            context_documents=final_docs,
            system_prompt=system_prompt,
            question_type=question_type
        )
        
        # 8. Format response
        return {
            "query": query_text,
            "normalized_query": normalized_query,
            "entities": entities,
            "answer": answer,
            # For evaluation, expose the broader pre-rerank results
            "retrieved_documents": retrieved_pre_rerank,
            # Also include reranked and final (MMR-selected) sets
            "reranked_documents": reranked_docs,
            "final_documents": final_docs,
            "run_manifest_id": run_manifest_id,
            "metadata": {
                "num_retrieved": len(retrieved_pre_rerank),
                "num_reranked": len(reranked_docs),
                "num_final": len(final_docs)
            }
        }
