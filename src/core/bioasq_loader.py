"""
BioASQ data loader for the Synergy task
Handles testset, golden, and feedback files
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class BioASQDataLoader:
    """Loader for BioASQ Synergy task data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize BioASQ data loader
        
        Args:
            data_dir: Directory containing BioASQ data files
        """
        self.data_dir = Path(data_dir)
    
    def load_testset(self, round_num: int) -> Dict[str, Any]:
        """
        Load testset for a specific round
        
        Args:
            round_num: Round number (1-4)
        
        Returns:
            Dictionary with questions
        """
        filepath = self.data_dir / f"testset_{round_num}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_golden(self, round_num: int) -> Dict[str, Any]:
        """
        Load golden (ground truth) data for a specific round
        
        Args:
            round_num: Round number (1-4)
        
        Returns:
            Dictionary with questions, documents, and snippets
        """
        filepath = self.data_dir / f"golden_round_{round_num}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_feedback(self, round_num: int) -> Dict[str, Any]:
        """
        Load feedback data accompanying a specific round
        
        Args:
            round_num: Round number (1-4)
        
        Returns:
            Dictionary with feedback from previous submissions
        """
        filepath = self.data_dir / f"feedback_accompanying_round_{round_num}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_questions_by_type(
        self, 
        data: Dict[str, Any], 
        question_type: str = None,
        answer_ready_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Filter questions by type and answer readiness
        
        Args:
            data: Loaded BioASQ data
            question_type: Filter by type (yesno, factoid, list, summary)
            answer_ready_only: Only return answerReady questions
        
        Returns:
            Filtered list of questions
        """
        questions = data.get("questions", [])
        
        filtered = []
        for q in questions:
            # Filter by answer readiness
            if answer_ready_only and not q.get("answerReady", False):
                continue
            
            # Filter by type
            if question_type and q.get("type") != question_type:
                continue
            
            filtered.append(q)
        
        return filtered
    
    def get_question_by_id(
        self, 
        data: Dict[str, Any], 
        question_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific question by ID
        
        Args:
            data: Loaded BioASQ data
            question_id: Question ID
        
        Returns:
            Question dictionary or None
        """
        questions = data.get("questions", [])
        for q in questions:
            if q.get("id") == question_id:
                return q
        return None
    
    def extract_document_ids(self, question: Dict[str, Any]) -> List[str]:
        """
        Extract PubMed document IDs from a question
        
        Args:
            question: Question dictionary
        
        Returns:
            List of PubMed IDs (as strings)
        """
        return question.get("documents", [])
    
    def extract_snippets(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract snippets from a question
        
        Args:
            question: Question dictionary
        
        Returns:
            List of snippet dictionaries with text, document, offsets
        """
        return question.get("snippets", [])
    
    def create_retrieval_corpus(
        self, 
        golden_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a corpus of documents from golden data
        Maps PubMed ID -> {snippets, question context}
        
        Args:
            golden_data: Loaded golden data
        
        Returns:
            Dictionary mapping doc_id to document information
        """
        corpus = {}
        
        for question in golden_data.get("questions", []):
            doc_ids = self.extract_document_ids(question)
            snippets = self.extract_snippets(question)
            
            # Group snippets by document
            for snippet in snippets:
                doc_id = snippet.get("document")
                if doc_id not in corpus:
                    corpus[doc_id] = {
                        "doc_id": doc_id,
                        "snippets": [],
                        "related_questions": []
                    }
                
                corpus[doc_id]["snippets"].append({
                    "text": snippet.get("text", ""),
                    "section": snippet.get("beginSection", ""),
                    "offset_start": snippet.get("offsetInBeginSection", 0),
                    "offset_end": snippet.get("offsetInEndSection", 0)
                })
                
                corpus[doc_id]["related_questions"].append({
                    "id": question.get("id"),
                    "body": question.get("body"),
                    "type": question.get("type")
                })
        
        return corpus
    
    def format_for_evaluation(
        self, 
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format question for evaluation
        
        Args:
            question: Question from golden data
        
        Returns:
            Dictionary with ground truth for evaluation
        """
        return {
            "question_id": question.get("id"),
            "question_text": question.get("body"),
            "question_type": question.get("type"),
            "relevant_docs": question.get("documents", []),
            "relevant_snippets": [s.get("text") for s in question.get("snippets", [])],
            "ideal_answer": question.get("ideal_answer", []),
            "exact_answer": question.get("exact_answer", "")
        }
