"""
Medical RAG Pipeline - FAISS-only variant
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
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.openai_client import OpenAIClient
from src.llm.stub_llm import StubLLM


class MedicalRAGPipelineFAISS:
    """End-to-end Medical RAG Pipeline using only FAISS dense retrieval"""

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
            device=encoder_config.get("device", "cpu")
        )

        faiss_config = self.config.get("faiss", {})
        self.faiss_index = FAISSIndex(
            index_path=faiss_config.get("save_path"),
            embedding_dim=encoder_config.get("embedding_dim", 768)
        )

        reranker_config = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            batch_size=reranker_config.get("batch_size", 16)
        )

        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "openai")
        if llm_provider == "stub" or os.getenv("LLM_PROVIDER") == "stub":
            self.llm = StubLLM()
        else:
            self.llm = OpenAIClient(
                model=llm_config.get("model", "gpt-4"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024),
                prompt_for_key=llm_config.get("prompt_for_key", True),
                use_keyring=llm_config.get("use_keyring", True),
                save_to_keyring=llm_config.get("save_to_keyring", False)
            )

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Build dense FAISS index only"""
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
        entities_text = []
        try:
            entities_text = [e.get("text", "").strip() for e in self.ner.extract_entities(normalized_query)]
            entities_text = [t for t in entities_text if t]
        except Exception:
            entities_text = []
        if faiss_append and entities_text:
            augmented_query = (normalized_query + " " + " ".join(entities_text[:faiss_max_entities])).strip()
        else:
            augmented_query = normalized_query
        query_embedding = self.encoder.encode_query(augmented_query)

        retrieval_cfg = self.config.get("retrieval", {})
        top_k_final = retrieval_cfg.get("top_k_final", 50)

        # FAISS retrieval only
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k_final)

        # Enrich docs from in-memory store
        retrieved_docs = []
        if dense_indices is not None and len(dense_indices) > 0:
            for score, idx in zip(dense_scores, dense_indices):
                if idx is None or (isinstance(idx, int) and idx < 0):
                    continue
                if not hasattr(self, "_doc_store") or idx >= len(self._doc_store):
                    continue
                d = self._doc_store[int(idx)]
                retrieved_docs.append({
                    "doc_id": d.get("doc_id"),
                    "score": float(score),
                    "dense_score": float(score),
                    "sparse_score": 0.0,
                    "index": int(idx),
                    "title": d.get("title", ""),
                    "abstract": d.get("abstract", ""),
                    "pub_date": d.get("pub_date")
                })

        # Rerank
        reranker_cfg = self.config.get("reranker", {})
        reranked_docs = self.reranker.rerank(
            query=normalized_query,
            documents=retrieved_docs,
            top_k=reranker_cfg.get("top_k", 20)
        )

        # MMR
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

        # Answer
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
