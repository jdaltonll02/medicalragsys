"""
Medical RAG Pipeline - BioBERT retriever variant (dense-only)
"""

import os
import uuid
from typing import Dict, Any, List

from src.core.normalizer import normalize_medical_query
from src.core.mmr import compute_mmr, compute_recency_scores
from src.core.utils import set_random_seed
from src.ner.ner_service import NERService
from src.encoder.biobert_encoder import BioBERTEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.biobert_retriever import BioBERTRetriever
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.openai_client import OpenAIClient
from src.llm.stub_llm import StubLLM


class MedicalRAGPipelineBioBERT:
    """End-to-end Medical RAG Pipeline using BioBERT dense retriever"""

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
        self.encoder = BioBERTEncoder(
            model_name=encoder_config.get("model", "dmis-lab/biobert-base-cased-v1.1"),
            device=encoder_config.get("device", "auto")
        )

        faiss_config = self.config.get("faiss", {})
        self.faiss_index = FAISSIndex(
            index_path=faiss_config.get("save_path"),
            embedding_dim=encoder_config.get("embedding_dim", 768)
        )
        self.biobert_retriever = BioBERTRetriever(self.faiss_index)

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
        """Build dense FAISS index with BioBERT embeddings"""
        if not documents:
            return
        self._doc_store = documents
        abstracts = [doc.get("abstract", "") for doc in documents]
        embeddings = self.encoder.encode(abstracts)
        self.faiss_index.add_vectors(embeddings)
        try:
            self.faiss_index.set_doc_ids([doc.get("doc_id") for doc in documents])
        except Exception:
            pass

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

        # Optionally append top entities for dense retrieval signal
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

        top_k_final = retrieval_cfg.get("top_k_final", 50)

        # BioBERT dense retrieval
        dense_results = self.biobert_retriever.retrieve(query_embedding, top_k=top_k_final)

        # Enrich documents from in-memory store
        retrieved_docs: List[Dict[str, Any]] = []
        for item in dense_results:
            idx = item.get("index")
            if isinstance(idx, int) and hasattr(self, "_doc_store") and 0 <= idx < len(self._doc_store):
                store_doc = self._doc_store[idx]
                retrieved_docs.append({
                    "doc_id": item.get("doc_id"),
                    "score": item.get("score"),
                    "dense_score": item.get("dense_score"),
                    "sparse_score": 0.0,
                    "index": idx,
                    "title": store_doc.get("title", ""),
                    "abstract": store_doc.get("abstract", ""),
                    "pub_date": store_doc.get("pub_date")
                })

        # Rerank
        reranker_cfg = self.config.get("reranker", {})
        reranked_docs = self.reranker.rerank(
            query=normalized_query,
            documents=retrieved_docs,
            top_k=reranker_cfg.get("top_k", 20)
        )

        # MMR for diversity
        if use_mmr and len(reranked_docs) > 0:
            mmr_cfg = self.config.get("mmr", {})
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

        # LLM Answer
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
            "retrieved_documents": final_docs,
            "run_manifest_id": run_manifest_id,
            "metadata": {
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(reranked_docs),
                "num_final": len(final_docs)
            }
        }
