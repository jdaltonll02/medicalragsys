"""
Stub LLM for testing and CI (deterministic responses)
"""

from typing import List, Dict, Any, Optional


class StubLLM:
    """Stub LLM that returns deterministic responses for testing"""
    
    def __init__(self, response_template: str = "This is a stub response to: {query}"):
        """
        Initialize stub LLM
        
        Args:
            response_template: Template for generating responses
        """
        self.response_template = response_template
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate deterministic stub response
        
        Args:
            prompt: User prompt
            system_prompt: Ignored for stub
            temperature: Ignored for stub
            max_tokens: Ignored for stub
        
        Returns:
            Stub response
        """
        return self.response_template.format(query=prompt[:100])
    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: str
    ) -> str:
        """
        Generate stub response with context
        
        Args:
            query: User query
            context_documents: Retrieved documents
            system_prompt: System prompt
        
        Returns:
            Stub response with citations
        """
        # Create response with document citations
        citations = ", ".join([doc.get("doc_id", "unknown") for doc in context_documents[:3]])
        
        response = (
            f"Based on the provided documents ({citations}), "
            f"here is a response to your query: {query[:50]}... "
            f"This is a deterministic stub response for testing purposes. "
            f"Retrieved {len(context_documents)} documents."
        )
        
        return response
