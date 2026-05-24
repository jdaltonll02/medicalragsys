"""
Medical RAG Pipeline - Hybrid MedCPT (dense) + BM25 (sparse)
"""

import os
import uuid
from typing import Dict, Any, List

from src.core.normalizer import normalize_medical_query
from src.core.mmr import compute_mmr, compute_recency_scores
from src.core.utils import set_random_seed
from src.ner.ner_service import NERService
from src.encoder.medcpt_encoder import MedCPTEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_medcpt_retriever import HybridMedCPTRetriever
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.openai_client import OpenAIClient
from src.llm.stub_llm import StubLLM


class MedicalRAGPipelineHybridMedCPT:
    """End-to-end Medical RAG Pipeline using MedCPT+BM25 hybrid retrieval"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        seed = config.get("pipeline", {}).get("seed", 42)
        set_random_seed(seed)
        self._initialize_components()

    def _initialize_components(self):
        ner_config = self.config.get("ner", {})
        self.ner = NERService(
            model_name=ner_config.get("model", "en_core_sci_sm"),
            confidence_threshold=ner_config.get("confidence_threshold", 0.7)
        )

        encoder_config = self.config.get("encoder", {})
        self.encoder = MedCPTEncoder(
            model_name=encoder_config.get("model", "ncbi/MedCPT-Query-Encoder"),
            device=encoder_config.get("device", "auto")
        )

        faiss_config = self.config.get("faiss", {})
        self.faiss_index = FAISSIndex(
            index_path=faiss_config.get("save_path"),
            embedding_dim=encoder_config.get("embedding_dim", 768)
        )

        bm25_config = self.config.get("bm25", {})
        self.bm25_retriever = BM25Retriever(
            host=bm25_config.get("elasticsearch_host", "localhost"),
            port=bm25_config.get("elasticsearch_port", 9200),
            index_name=bm25_config.get("index_name", "medical_docs")
        )

        self.hybrid_retriever = HybridMedCPTRetriever(
            faiss_index=self.faiss_index,
            bm25_retriever=self.bm25_retriever,
            alpha=0.5
        )

        reranker_config = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            batch_size=reranker_config.get("batch_size", 16),
            device=reranker_config.get("device", "auto")
        )

        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "openai")
        if llm_provider == "stub" or os.getenv("LLM_PROVIDER") == "stub":
            self.llm = StubLLM()
        else:
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

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Build indices for FAISS (dense) and BM25 (sparse)"""
        if not documents:
            return
        self._doc_store = documents
        # Encode abstracts for FAISS
        abstracts = [doc.get("abstract", "") for doc in documents]
        embeddings = self.encoder.encode(abstracts)
        self.faiss_index.add_vectors(embeddings)
        try:
            self.faiss_index.set_doc_ids([doc.get("doc_id") for doc in documents])
        except Exception:
            pass
        # Index documents into Elasticsearch
        try:
            try:
                self.bm25_retriever.reset_index()
            except Exception:
                pass
            self.bm25_retriever.index_documents(documents)
        except Exception as e:
            print(f"Warning: BM25 indexing failed: {e}")

    def process_query(
        self,
        query_text: str,
        top_k: int = 10,
        use_mmr: bool = True,
        recency_boost: bool = True
    ) -> Dict[str, Any]:
        run_manifest_id = str(uuid.uuid4())
        normalized_query = normalize_medical_query(query_text)
        entities = self.ner.extract_entities(normalized_query)

        # Encode query (optionally append entities)
        retrieval_cfg = self.config.get("retrieval", {})
        faiss_append = retrieval_cfg.get("faiss_entity_append", True)
        faiss_max_entities = retrieval_cfg.get("faiss_max_entities", 3)
        ent_texts = [e.get("text", "").strip() for e in entities if isinstance(e, dict)]
        ent_texts = [t for t in ent_texts if t]
        if faiss_append and ent_texts:
            augmented_query = (normalized_query + " " + " ".join(ent_texts[:faiss_max_entities])).strip()
        else:
            augmented_query = normalized_query
        query_embedding = self.encoder.encode_query(augmented_query)

        # Hybrid retrieval
        retrieved_docs = self.hybrid_retriever.retrieve(
            query=normalized_query,
            query_embedding=query_embedding,
            top_k_dense=retrieval_cfg.get("top_k_dense", 100),
            top_k_sparse=retrieval_cfg.get("top_k_sparse", 100),
            top_k_final=retrieval_cfg.get("top_k_final", 50),
            entities=entities,
            entity_boost=retrieval_cfg.get("bm25_entity_boost", 2.0),
            max_entities=retrieval_cfg.get("bm25_max_entities", 5),
        )

        # Enrich retrieved docs
        enriched = []
        for item in retrieved_docs:
            doc = {
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "index": item.get("index")
            }
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
        retrieved_pre_rerank = list(enriched)

        # Rerank
        reranker_cfg = self.config.get("reranker", {})
        reranked_docs = self.reranker.rerank(
            query=normalized_query,
            documents=retrieved_pre_rerank,
            top_k=reranker_cfg.get("top_k", 20)
        )

        # MMR
        mmr_cfg = self.config.get("mmr", {})
        if use_mmr and mmr_cfg.get("enabled", True) and len(reranked_docs) > 0:
            doc_embeddings = self.encoder.encode([d.get("abstract", "") for d in reranked_docs])
            recency_scores = None
            if recency_boost:
                pub_dates = [d.get("pub_date", "2000-01-01") for d in reranked_docs]
                recency_scores = compute_recency_scores(pub_dates)
            selected = compute_mmr(
                query_embedding=query_embedding,
                candidate_embeddings=doc_embeddings,
                lambda_param=mmr_cfg.get("lambda_param", 0.7),
                top_k=top_k,
                recency_scores=recency_scores,
                recency_weight=mmr_cfg.get("recency_weight", 0.3) if recency_boost else 0.0
            )
            final_docs = [reranked_docs[i] for i in selected]
        else:
            final_docs = reranked_docs[:top_k]

        # LLM
        llm_cfg = self.config.get("llm", {})
        system_prompt = llm_cfg.get("system_prompt", "You are a medical AI assistant.")
        answer = self.llm.generate_with_context(
            query=query_text,
            context_documents=final_docs,
            system_prompt=system_prompt
        )

        return {
            "query": query_text,
            "normalized_query": normalized_query,
            "entities": entities,
            "answer": answer,
            "retrieved_documents": retrieved_pre_rerank,
            "reranked_documents": reranked_docs,
            "final_documents": final_docs,
            "run_manifest_id": run_manifest_id,
            "metadata": {
                "num_retrieved": len(retrieved_pre_rerank),
                "num_reranked": len(reranked_docs),
                "num_final": len(final_docs)
            }
        }
