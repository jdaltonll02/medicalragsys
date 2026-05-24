"""
BioASQ Synergy 2026 Pipeline - End-to-end system for Synergy task
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.core.synergy_formatter import SynergyFormatter, SnippetExtractor, FeedbackLoader
from src.core.answer_generator import AnswerGenerator, YesNoAnswerGenerator, SummaryAnswerGenerator
from src.core.normalizer import normalize_medical_query


logger = logging.getLogger(__name__)


class SynergyPipeline:
    """Complete Synergy 2026 pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Synergy pipeline
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.llm_client = None
        self.retrieval_pipeline = None
    
    def set_llm_client(self, llm_client):
        """Set LLM client for answer generation"""
        self.llm_client = llm_client
    
    def set_retrieval_pipeline(self, retrieval_pipeline):
        """Set retrieval pipeline"""
        self.retrieval_pipeline = retrieval_pipeline
    
    def process_round(
        self,
        testset_path: str,
        round_num: int,
        output_dir: str,
        feedback_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a complete Synergy round
        
        Args:
            testset_path: Path to testset JSON
            round_num: Round number (1-4)
            output_dir: Output directory for results
            feedback_path: Path to feedback from previous round (for rounds 2+)
        
        Returns:
            Dict with 'submission' and 'metrics'
        """
        logger.info(f"Processing Synergy round {round_num}")
        
        # Load test questions
        with open(testset_path, 'r') as f:
            testset = json.load(f)
        
        questions = testset.get("questions", [])
        logger.info(f"Loaded {len(questions)} questions")
        
        # Load feedback if available; build exclusion sets per question
        golden_snippets = {}
        golden_docs = {}
        golden_answers = {}
        excluded_docs: dict = {}   # question_id -> set of doc_ids already seen

        if feedback_path and Path(feedback_path).exists():
            logger.info(f"Loading feedback from {feedback_path}")
            feedback = FeedbackLoader.load_feedback(feedback_path)
            golden_snippets = FeedbackLoader.extract_golden_snippets(feedback)
            golden_docs = FeedbackLoader.extract_golden_documents(feedback)
            golden_answers = FeedbackLoader.extract_golden_answers(feedback)
            # All documents in the feedback (golden or not) must not be resubmitted
            for fbq in feedback.get("questions", []):
                q_id = fbq.get("id")
                seen = {str(d.get("id")) for d in fbq.get("documents", [])}
                if seen:
                    excluded_docs[q_id] = seen
            logger.info(f"Loaded feedback for {len(golden_docs)} questions")
        
        # Process each question
        predictions = []
        exact_answers = []
        ideal_answers = []
        
        for i, question in enumerate(questions):
            q_id = question.get("id")
            logger.info(f"Processing question {i+1}/{len(questions)}: {q_id}")
            
            # Retrieve documents, then exclude any already seen in feedback
            query = question.get("body", "")
            retrieved_docs = self._retrieve_documents(query)
            if q_id in excluded_docs:
                retrieved_docs = [
                    d for d in retrieved_docs
                    if str(d.get("doc_id")) not in excluded_docs[q_id]
                ]
            
            # For non-answer-ready questions, skip answer generation
            # For answer-ready questions, generate answers
            is_answer_ready = question.get("answerReady", False)
            
            exact_ans = []
            ideal_ans = ""
            
            if is_answer_ready:
                q_type = question.get("type", "factoid")
                
                # Generate appropriate answer type
                if q_type == "yesno":
                    ideal_ans = YesNoAnswerGenerator.generate_yesno_answer(
                        question, retrieved_docs, self.llm_client
                    )
                    exact_ans = [[ideal_ans]] if ideal_ans else []
                
                elif q_type == "factoid":
                    exact_ans = AnswerGenerator.generate_exact_answer(
                        question, retrieved_docs, self.llm_client
                    )
                    ideal_ans = AnswerGenerator.generate_ideal_answer(
                        question, retrieved_docs, self.llm_client
                    )
                
                elif q_type == "list":
                    exact_ans = AnswerGenerator.generate_exact_answer(
                        question, retrieved_docs, self.llm_client
                    )
                    ideal_ans = AnswerGenerator.generate_ideal_answer(
                        question, retrieved_docs, self.llm_client
                    )
                
                elif q_type == "summary":
                    ideal_ans = SummaryAnswerGenerator.generate_summary(
                        question, retrieved_docs, self.llm_client
                    )
            
            # Store results
            predictions.append({
                "question_id": q_id,
                "retrieved_documents": retrieved_docs
            })
            exact_answers.append(exact_ans)
            ideal_answers.append(ideal_ans)
        
        # Format as Synergy submission
        submission = SynergyFormatter.format_submission(
            questions, predictions, exact_answers, ideal_answers
        )
        
        # Save submission
        output_path = Path(output_dir) / f"synergy_round_{round_num}_submission.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        SynergyFormatter.save_submission(submission, str(output_path))
        logger.info(f"Saved submission to {output_path}")
        
        return {
            "submission": submission,
            "metrics": {
                "num_questions": len(questions),
                "num_answer_ready": sum(1 for q in questions if q.get("answerReady")),
                "output_path": str(output_path)
            }
        }
    
    def _retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve and enrich relevant documents for a query."""
        if self.retrieval_pipeline is None:
            return []

        retrieval_cfg = self.config.get("retrieval", {})
        if top_k is None:
            top_k = retrieval_cfg.get("top_k_final", 50)

        try:
            normalized = normalize_medical_query(query)

            # Extract NER entities for entity-boosted BM25
            entities = []
            if hasattr(self.retrieval_pipeline, "ner"):
                try:
                    entities = self.retrieval_pipeline.ner.extract_entities(normalized)
                except Exception:
                    pass

            # Encode query with the dedicated query encoder
            query_embedding = self.retrieval_pipeline.encoder.encode_query(normalized)

            raw_docs = self.retrieval_pipeline.hybrid_retriever.retrieve(
                query=normalized,
                query_embedding=query_embedding,
                top_k_dense=retrieval_cfg.get("top_k_dense", 100),
                top_k_sparse=retrieval_cfg.get("top_k_sparse", 100),
                top_k_final=top_k,
                entities=entities,
                entity_boost=retrieval_cfg.get("bm25_entity_boost", 2.0),
                max_entities=retrieval_cfg.get("bm25_max_entities", 5),
            )

            # Enrich with title/abstract from BM25 source or in-memory doc store
            enriched = []
            for item in raw_docs:
                doc = {
                    "doc_id": item.get("doc_id"),
                    "score": item.get("score"),
                    "dense_score": item.get("dense_score"),
                    "sparse_score": item.get("sparse_score"),
                    "index": item.get("index"),
                }
                src = item.get("source")
                if src and isinstance(src, dict):
                    doc["title"] = src.get("title", "")
                    doc["abstract"] = src.get("abstract", "")
                    doc["pub_date"] = src.get("pub_date")
                else:
                    idx = doc.get("index")
                    if (
                        isinstance(idx, int)
                        and hasattr(self.retrieval_pipeline, "_doc_store")
                        and 0 <= idx < len(self.retrieval_pipeline._doc_store)
                    ):
                        store_doc = self.retrieval_pipeline._doc_store[idx]
                        doc["title"] = store_doc.get("title", "")
                        doc["abstract"] = store_doc.get("abstract", "")
                        doc["pub_date"] = store_doc.get("pub_date")
                enriched.append(doc)

            return enriched
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


class SynergyEvaluator:
    """Evaluate Synergy submissions against golden answers"""
    
    @staticmethod
    def load_submission(path: str) -> Dict[str, Any]:
        """Load submission JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def evaluate_snippets(
        submitted_snippets: List[Dict[str, Any]],
        golden_snippets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate snippet retrieval
        
        Args:
            submitted_snippets: Submitted snippets
            golden_snippets: Golden snippets
        
        Returns:
            Dict with precision, recall, F1
        """
        submitted_texts = set(s.get("text", "")[:100] for s in submitted_snippets)
        golden_texts = set(s.get("text", "")[:100] for s in golden_snippets)
        
        if not golden_texts:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        true_positives = len(submitted_texts & golden_texts)
        false_positives = len(submitted_texts - golden_texts)
        false_negatives = len(golden_texts - submitted_texts)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    @staticmethod
    def evaluate_documents(
        submitted_docs: List[str],
        golden_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate document retrieval
        
        Args:
            submitted_docs: Submitted document PMIDs
            golden_docs: Golden document PMIDs
        
        Returns:
            Dict with precision, recall, F1, MRR
        """
        submitted_set = set(str(d) for d in submitted_docs)
        golden_set = set(str(d) for d in golden_docs)
        
        if not golden_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mrr": 1.0}
        
        # Metrics
        true_positives = len(submitted_set & golden_set)
        false_positives = len(submitted_set - golden_set)
        false_negatives = len(golden_set - submitted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR: reciprocal rank of the first relevant document
        mrr = 0.0
        for i, doc in enumerate(submitted_docs):
            if str(doc) in golden_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {"precision": precision, "recall": recall, "f1": f1, "mrr": mrr}
    
    @staticmethod
    def evaluate_answers(
        submitted_answer: str,
        golden_answers: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate answer quality (simplified)
        
        Args:
            submitted_answer: Submitted answer
            golden_answers: List of acceptable golden answers
        
        Returns:
            Dict with exact_match, approximate_match
        """
        if not golden_answers:
            return {"exact_match": 0.0, "approximate_match": 0.0}
        
        submitted_lower = str(submitted_answer).lower().strip()
        
        # Exact match
        for golden in golden_answers:
            if submitted_lower == str(golden).lower().strip():
                return {"exact_match": 1.0, "approximate_match": 1.0}
        
        # Approximate match (substring)
        for golden in golden_answers:
            golden_lower = str(golden).lower().strip()
            if golden_lower in submitted_lower or submitted_lower in golden_lower:
                return {"exact_match": 0.0, "approximate_match": 1.0}
        
        return {"exact_match": 0.0, "approximate_match": 0.0}
