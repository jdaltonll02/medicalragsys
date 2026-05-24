"""
Google Gemini LLM client
"""

from typing import List, Dict, Any, Optional
import os
import sys
import json


class GeminiClient:
    """Google Gemini API client for LLM generation"""
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize Gemini client
        
        Args:
            model: Model name (e.g., gemini-2.0-flash, gemini-1.5-pro)
            api_key: Google API key (or use GOOGLE_API_KEY env var)
            project_id: Google Cloud project ID (for some APIs)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        # Resolve API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        sys.stderr.write(f"[GEMINI] Initializing with:\n")
        sys.stderr.write(f"  - Model: {self.model}\n")
        sys.stderr.write(f"  - API Key: {'***' if self.api_key else 'NOT SET'}\n")
        sys.stderr.write(f"  - Project ID: {self.project_id or 'none'}\n")
        sys.stderr.flush()
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            
            if not self.api_key:
                raise RuntimeError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            
            sys.stderr.write("[GEMINI] Client initialized successfully\n")
            sys.stderr.flush()
        except ImportError:
            sys.stderr.write("[GEMINI] ERROR: google-generativeai package not installed\n")
            sys.stderr.write("  Install with: pip install google-generativeai\n")
            sys.stderr.flush()
            raise
        except Exception as e:
            sys.stderr.write(f"[GEMINI] Failed to initialize: {e}\n")
            sys.stderr.flush()
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from Gemini using generative API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions)
            temperature: Sampling temperature override
            max_tokens: Maximum tokens override
        
        Returns:
            Generated text response
        """
        if self.client is None:
            return "Error: Gemini client not initialized"
        
        try:
            # Build the full prompt with system instructions
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate content
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature or self.temperature,
                    "max_output_tokens": max_tokens or self.max_tokens,
                }
            )
            
            # Extract generated text
            if response and response.text:
                return response.text
            else:
                return "Error: Empty response from Gemini"
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: str
    ) -> str:
        """
        Generate response with retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved documents with citations
            system_prompt: System prompt
        
        Returns:
            Generated answer
        """
        # Build context in JSON structure (matching OpenAI format)
        docs_obj: Dict[str, Any] = {}
        for i, doc in enumerate(context_documents, 1):
            relevance = doc.get("score")
            if relevance is None:
                relevance = doc.get("dense_score")
            try:
                relevance_val = float(relevance) if relevance is not None else 0.0
            except Exception:
                relevance_val = 0.0

            docs_obj[f"doc{i}"] = {
                "PMID": doc.get("doc_id"),
                "title": doc.get("title", ""),
                "content": doc.get("abstract", ""),
                "relevance_score": relevance_val
            }

        user_part = f"User Prompt: Answer the following question: {query}"
        context_part = "Context Prompt: Here are the documents:\n" + json.dumps(docs_obj, ensure_ascii=False, indent=2)
        full_prompt = user_part + "\n\n" + context_part

        return self.generate(full_prompt, system_prompt=system_prompt)
