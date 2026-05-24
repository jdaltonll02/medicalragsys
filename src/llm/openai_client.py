"""
OpenAI LLM client
"""

from typing import List, Dict, Any, Optional
import os
import sys
import json
# from dotenv import load_dotenv
import httpx

class OpenAIClient:
    """OpenAI API client for LLM generation"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        project_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        prompt_for_key: bool = True,
        use_keyring: bool = True,
        save_to_keyring: bool = False,
    ):
        """
        Initialize OpenAI client
        
        Args:
            model: Model name (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            base_url: API base URL (or use OPENAI_BASE_URL env var)
            project_id: OpenAI project ID (or use OPENAI_PROJECT_ID env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.project_id = project_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client(prompt_for_key=prompt_for_key, use_keyring=use_keyring, save_to_keyring=save_to_keyring)
    
    # def _initialize_client(self, prompt_for_key: bool = True, use_keyring: bool = True, save_to_keyring: bool = False):
    #     """Initialize OpenAI client"""
    #     if not self.api_key:
    #         # Optionally prompt the user for a key if running interactively
    #         if prompt_for_key and sys.stdin and sys.stdin.isatty():
    #             try:
    #                 import getpass
    #                 print("OpenAI API key is required. It will not be printed.")
    #                 entered = getpass.getpass("Paste OpenAI API key: ")
    #                 if entered:
    #                     self.api_key = entered.strip()
    #                     if save_to_keyring:
    #                         try:
    #                             import keyring  # type: ignore
    #                             keyring.set_password("medical_rag_system", "openai_api_key", self.api_key)
    #                         except Exception:
    #                             pass
    #             except Exception:
    #                 pass
    #         if not self.api_key:
    #             raise RuntimeError("OPENAI_API_KEY is not set — cannot initialize OpenAI client")
        
    #     try:
    #         from openai import OpenAI
    #         # Initialize basic client; avoid unsupported kwargs
    #         # if self.base_url and self.organization:
    #         #     self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, organization=self.organization)
    #         # elif self.base_url:
    #         #     self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    #         # elif self.organization:
    #         #     self.client = OpenAI(api_key=self.api_key, organization=self.organization)
    #         # else:
    #         #     self.client = OpenAI(api_key=self.api_key)
    #         self.client = OpenAI(api_key=self.api_key)
    #     except Exception as e:
    #         print("FATAL: OpenAI client initialization failed", file=sys.stderr)
    #         raise

    def _initialize_client(self, *_, **__):
        from openai import OpenAI

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or None

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
            )
            sys.stderr.write(f"[OPENAI] Client initialized (model={self.model}, base_url={base_url or 'default'})\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[OPENAI] Failed to initialize: {e}\n")
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
        Generate response from LLM using chat completions API.
        """
        if self.client is None:
            return "Error: OpenAI client not initialized"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use chat.completions.create() instead of responses.create()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            # Extract generated text
            return response.choices[0].message.content or ""

        except Exception as e:
            return f"Error generating response: {e}"

    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: str,
        question_type: Optional[str] = None
    ) -> str:
        """
        Generate response with retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved documents with citations
            system_prompt: System prompt
            question_type: Type of question (yesno, factoid, list, summary)
        
        Returns:
            Generated answer
        """
        # Build context in the requested JSON structure
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

        # Add question type guidance if provided
        type_guidance = ""
        if question_type:
            if question_type == "yesno":
                type_guidance = "\n\nThis is a YES/NO question. Start your answer with 'Yes' or 'No', followed by a brief explanation."
            elif question_type == "factoid":
                type_guidance = (
                    "\n\nThis is a FACTOID question. You MUST name a specific entity (gene, drug, disease, protein, number, etc.) as your answer. "
                    "Do NOT say 'information not available' or 'cannot be determined'. "
                    "Extract the best answer from the provided documents and state it directly at the start of your response."
                )
            elif question_type == "list":
                type_guidance = (
                    "\n\nThis is a LIST question. You MUST respond with ONLY a numbered list — no introduction, no conclusion, no prose. "
                    "Each line must be a single short entity name (drug, gene, disease, protein, etc.). "
                    "Format EXACTLY as:\n1. First item\n2. Second item\n3. Third item\n"
                    "Use the documents and your biomedical knowledge. Always provide a list — never leave it empty."
                )
            elif question_type == "summary":
                type_guidance = "\n\nThis is a SUMMARY question. Provide a comprehensive paragraph (2-3 sentences) summarizing the key information."

        user_part = f"User Prompt: Answer the following question: {query}{type_guidance}"
        context_part = "Context Prompt: Here are the documents:\n" + json.dumps(docs_obj, ensure_ascii=False, indent=2)
        full_prompt = user_part + "\n\n" + context_part

        return self.generate(full_prompt, system_prompt=system_prompt)
