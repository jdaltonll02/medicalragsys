"""
BioASQ Synergy 2026 utilities for snippet extraction and formatting
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import difflib


class SnippetExtractor:
    """Extract relevant snippets from abstracts with character offsets"""
    
    @staticmethod
    def find_snippet(text: str, query_terms: List[str], context_window: int = 150) -> Optional[Dict[str, Any]]:
        """
        Find a snippet in text that contains query terms with context
        
        Args:
            text: Abstract or document text
            query_terms: Terms to search for
            context_window: Characters before/after match to include
        
        Returns:
            Dict with text, offsetInBeginSection, offsetInEndSection or None
        """
        if not text or not query_terms:
            return None
        
        # Find sentence containing most query terms
        sentences = re.split(r'(?<=[.!?])\s+', text)
        best_match = None
        best_score = 0
        best_start = 0
        
        current_offset = 0
        for sentence in sentences:
            # Count matching terms in this sentence
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term.lower() in sentence_lower)
            
            if score > best_score:
                best_score = score
                best_match = sentence
                best_start = current_offset
            
            current_offset += len(sentence) + 1  # +1 for space
        
        if best_match and best_score > 0:
            # Record the original end offset before any truncation
            original_end = best_start + len(best_match)
            max_len = 250
            if len(best_match) > max_len:
                best_match = best_match[:max_len] + "..."

            return {
                "text": best_match,
                "offsetInBeginSection": best_start,
                "offsetInEndSection": original_end
            }
        
        return None
    
    @staticmethod
    def extract_snippets(
        query: str,
        documents: List[Dict[str, Any]],
        max_snippets: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract snippets from multiple documents for a query
        
        Args:
            query: Question text
            documents: List of documents with 'doc_id', 'title', 'abstract'
            max_snippets: Maximum snippets to extract
        
        Returns:
            List of snippet dicts with document, text, and offsets
        """
        # Parse query into terms
        query_terms = [t.strip() for t in query.split() if len(t) > 3]
        
        snippets = []
        for doc in documents[:max_snippets]:
            # Try abstract first
            abstract = doc.get("abstract", "")
            title = doc.get("title", "")
            snippet_info = None  # Initialize to None to avoid UnboundLocalError
            
            if abstract:
                snippet_info = SnippetExtractor.find_snippet(abstract, query_terms)
                if snippet_info:
                    snippet_info["document"] = str(doc.get("doc_id"))
                    snippet_info["beginSection"] = "abstract"
                    snippet_info["endSection"] = "abstract"
                    snippets.append(snippet_info)
            
            # Try title if no abstract snippet found
            if not snippet_info and title:
                snippet_info = SnippetExtractor.find_snippet(title, query_terms)
                if snippet_info:
                    snippet_info["document"] = str(doc.get("doc_id"))
                    snippet_info["beginSection"] = "title"
                    snippet_info["endSection"] = "title"
                    snippets.append(snippet_info)
        
        return snippets


class SynergyFormatter:
    """Format predictions in BioASQ Synergy 2026 format"""
    
    @staticmethod
    def format_submission(
        questions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        exact_answers: Optional[List[List[List[str]]]] = None,
        ideal_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert predictions to Synergy submission format
        
        Args:
            questions: List of question dicts with 'id', 'body', 'type', 'answerReady'
            predictions: List of prediction dicts with 'question_id', 'retrieved_documents'
            exact_answers: Optional list of exact answers (for answerReady=true)
            ideal_answers: Optional list of ideal answers (for answerReady=true)
        
        Returns:
            Dict in Synergy submission format
        """
        synergy_questions = []
        
        pred_map = {p.get("question_id"): p for p in predictions}
        exact_map = {q["id"]: exact_answers[i] if exact_answers and i < len(exact_answers) else [] 
                     for i, q in enumerate(questions)}
        ideal_map = {q["id"]: ideal_answers[i] if ideal_answers and i < len(ideal_answers) else ""
                     for i, q in enumerate(questions)}
        
        for question in questions:
            q_id = question["id"]
            pred = pred_map.get(q_id, {})
            
            # Extract documents
            retrieved_docs = pred.get("retrieved_documents", [])
            document_ids = [str(doc.get("doc_id")) for doc in retrieved_docs]
            
            # Extract snippets
            snippets = SnippetExtractor.extract_snippets(
                question.get("body", ""),
                retrieved_docs,
                max_snippets=10
            )
            
            # Build question response
            synergy_q = {
                "id": q_id,
                "body": question.get("body", ""),
                "type": question.get("type", ""),
                "documents": document_ids,
                "snippets": snippets,
                "answer_ready": question.get("answerReady", False)
            }
            
            # Add exact and ideal answers only if answerReady=true
            if question.get("answerReady", False):
                synergy_q["exact_answer"] = exact_map.get(q_id, [])
                synergy_q["ideal_answer"] = ideal_map.get(q_id, "")
            else:
                synergy_q["exact_answer"] = []
                synergy_q["ideal_answer"] = ""
            
            synergy_questions.append(synergy_q)
        
        return {"questions": synergy_questions}
    
    @staticmethod
    def save_submission(
        submission: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save submission to JSON file"""
        import json
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)


class FeedbackLoader:
    """Load feedback from previous rounds for iterative improvement"""
    
    @staticmethod
    def load_feedback(feedback_path: str) -> Dict[str, Any]:
        """Load feedback JSON file"""
        import json
        with open(feedback_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def extract_golden_snippets(feedback: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract golden snippets from feedback
        
        Returns:
            Dict mapping question_id to list of golden snippet texts
        """
        golden_snippets = {}
        
        for question in feedback.get("questions", []):
            q_id = question.get("id")
            q_snippets = []
            
            for snippet in question.get("snippets", []):
                if snippet.get("golden", False):
                    q_snippets.append(snippet.get("text", ""))
            
            if q_snippets:
                golden_snippets[q_id] = q_snippets
        
        return golden_snippets
    
    @staticmethod
    def extract_golden_documents(feedback: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract golden document PMIDs from feedback
        
        Returns:
            Dict mapping question_id to list of golden PMID strings
        """
        golden_docs = {}
        
        for question in feedback.get("questions", []):
            q_id = question.get("id")
            q_docs = []
            
            for doc in question.get("documents", []):
                if doc.get("golden", False):
                    q_docs.append(str(doc.get("id")))
            
            if q_docs:
                golden_docs[q_id] = q_docs
        
        return golden_docs
    
    @staticmethod
    def extract_golden_answers(feedback: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract golden exact/ideal answers from feedback
        
        Returns:
            Dict mapping question_id to dict with 'exact_answer' and 'ideal_answer'
        """
        golden_answers = {}
        
        for question in feedback.get("questions", []):
            q_id = question.get("id")
            
            exact = question.get("exact_answer")
            ideal = question.get("ideal_answer")
            
            if exact or ideal:
                golden_answers[q_id] = {
                    "exact_answer": exact if isinstance(exact, list) else [exact] if exact else [],
                    "ideal_answer": ideal if isinstance(ideal, list) else [ideal] if ideal else []
                }
        
        return golden_answers
