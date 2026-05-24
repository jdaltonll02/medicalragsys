"""
Answer generation for BioASQ Synergy questions
"""

from typing import List, Dict, Any, Optional
import json


class AnswerGenerator:
    """Generate exact and ideal answers for Synergy questions"""
    
    @staticmethod
    def extract_entities(text: str, entity_type: str = "generic") -> List[str]:
        """
        Extract entities from text (simplified entity extraction)
        
        Args:
            text: Text to extract entities from
            entity_type: Type of entity (gene, disease, drug, generic)
        
        Returns:
            List of extracted entities
        """
        # Simple implementation - can be enhanced with NER
        import re
        
        entities = []
        
        if entity_type == "gene":
            # Match common gene patterns (capitalized words)
            matches = re.findall(r'\b[A-Z]{2,}\b', text)
            entities = list(set(matches))
        elif entity_type == "disease":
            # Match disease-like terms
            keywords = ['disease', 'syndrome', 'disorder', 'cancer', 'diabetes']
            for keyword in keywords:
                if keyword in text.lower():
                    # Extract phrase around keyword
                    idx = text.lower().find(keyword)
                    start = max(0, idx - 30)
                    end = min(len(text), idx + 60)
                    phrase = text[start:end].strip()
                    if phrase:
                        entities.append(phrase)
        else:
            # Generic: extract capitalized phrases
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities = list(set(matches))[:10]
        
        return entities
    
    @staticmethod
    def generate_exact_answer(
        question: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
        llm_judge=None
    ) -> List[List[str]]:
        """
        Generate exact answer for factoid/list questions
        
        Args:
            question: Question dict with 'body' and 'type'
            retrieved_documents: List of retrieved documents
            llm_judge: Optional LLM for answer extraction
        
        Returns:
            List of [answer] pairs
        """
        q_type = question.get("type", "factoid")
        
        # Extract potential answers from documents
        candidates = []
        
        for doc in retrieved_documents[:5]:
            abstract = doc.get("abstract", "")
            title = doc.get("title", "")
            text = f"{title}. {abstract}"
            
            # Extract entities based on question type
            if q_type == "factoid":
                entities = AnswerGenerator.extract_entities(text, "gene")
                if not entities:
                    entities = AnswerGenerator.extract_entities(text, "generic")
                candidates.extend(entities)
            elif q_type == "list":
                entities = AnswerGenerator.extract_entities(text, "disease")
                if not entities:
                    entities = AnswerGenerator.extract_entities(text, "generic")
                candidates.extend(entities)
        
        # Format as required (list of [entity] pairs)
        exact_answers = [[c] for c in list(set(candidates[:5]))]
        
        return exact_answers if exact_answers else [[]]
    
    @staticmethod
    def generate_ideal_answer(
        question: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
        llm_client=None
    ) -> str:
        """
        Generate ideal answer (paragraph summary)
        
        Args:
            question: Question dict
            retrieved_documents: List of retrieved documents
            llm_client: Optional LLM client for generation
        
        Returns:
            Ideal answer text
        """
        if llm_client is None or not (hasattr(llm_client, "client") and llm_client.client):
            # Fallback: generate from abstracts
            abstracts = [doc.get("abstract", "")[:200] for doc in retrieved_documents[:3]]
            abstracts = [a for a in abstracts if a]
            return " ".join(abstracts) if abstracts else ""
        
        # Use LLM to generate answer
        try:
            q_type = question.get('type', 'factoid')
            
            if q_type == 'factoid':
                system_prompt = """You are a biomedical expert. Answer factoid questions with the specific entity (gene, drug, disease, etc.) first, followed by a brief explanation.
Format: Start with the main answer entity, then provide 1-2 sentences of context."""
            elif q_type == 'list':
                system_prompt = """You are a biomedical expert. Answer list questions with a numbered list.
Format:
1. First item
2. Second item
3. Third item
(etc.)
Follow the list with a brief 1-sentence summary if needed."""
            else:
                system_prompt = "Generate a concise paragraph-length answer to the biomedical question based on the provided documents."
            
            context = "\n".join([
                f"Document {i+1} ({doc.get('doc_id')}): {doc.get('abstract', '')[:300]}"
                for i, doc in enumerate(retrieved_documents[:5])
            ])
            
            prompt = f"""Question: {question.get('body')}

Retrieved documents:
{context}

Generate a concise answer (1-3 sentences):"""
            
            answer = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            return answer if answer else ""
        except Exception as e:
            print(f"Error generating ideal answer: {e}")
            abstracts = [doc.get("abstract", "")[:200] for doc in retrieved_documents[:3]]
            abstracts = [a for a in abstracts if a]
            return " ".join(abstracts) if abstracts else ""


class YesNoAnswerGenerator:
    """Generate yes/no answers for yes/no questions"""
    
    @staticmethod
    def generate_yesno_answer(
        question: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
        llm_client=None
    ) -> str:
        """
        Generate yes/no answer
        
        Args:
            question: Question dict
            retrieved_documents: List of retrieved documents
            llm_client: Optional LLM client
        
        Returns:
            "yes", "no", or ""
        """
        if llm_client is None or not (hasattr(llm_client, "client") and llm_client.client):
            return ""
        
        try:
            system_prompt = """You are a biomedical expert answering yes/no questions.
Based on the evidence provided, answer with ONLY 'yes' or 'no' as the first word.
Do NOT include any explanation, just the word 'yes' or 'no'."""
            
            context = "\n".join([
                f"- {doc.get('abstract', '')[:200]}"
                for doc in retrieved_documents[:3]
            ])
            
            prompt = f"""Question: {question.get('body')}

Evidence:
{context}

Answer with ONLY 'yes' or 'no':"""
            
            answer = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=10
            )
            
            answer_lower = answer.lower().strip()
            if "yes" in answer_lower:
                return "yes"
            elif "no" in answer_lower:
                return "no"
            else:
                return ""
        except Exception as e:
            print(f"Error generating yes/no answer: {e}")
            return ""


class SummaryAnswerGenerator:
    """Generate summary answers for summary questions"""
    
    @staticmethod
    def generate_summary(
        question: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
        llm_client=None
    ) -> str:
        """
        Generate summary answer
        
        Args:
            question: Question dict
            retrieved_documents: List of retrieved documents
            llm_client: Optional LLM client
        
        Returns:
            Summary text
        """
        if llm_client is None or not (hasattr(llm_client, "client") and llm_client.client):
            # Fallback: combine abstracts
            abstracts = [doc.get("abstract", "") for doc in retrieved_documents[:3]]
            abstracts = [a for a in abstracts if a]
            return " ".join(abstracts) if abstracts else ""
        
        try:
            system_prompt = """You are a biomedical expert. Generate a comprehensive, well-structured summary paragraph.
Provide key information in 2-4 sentences. Be concise but complete."""
            
            context = "\n".join([
                f"Document {i+1}: {doc.get('abstract', '')[:300]}"
                for i, doc in enumerate(retrieved_documents[:5])
            ])
            
            prompt = f"""Question: {question.get('body')}

Relevant documents:
{context}

Generate a comprehensive summary paragraph (2-4 sentences):"""
            
            answer = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            return answer if answer else ""
        except Exception as e:
            print(f"Error generating summary: {e}")
            abstracts = [doc.get("abstract", "") for doc in retrieved_documents[:3]]
            abstracts = [a for a in abstracts if a]
            return " ".join(abstracts) if abstracts else ""
