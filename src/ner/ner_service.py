"""
NER service for biomedical entity extraction
"""

from typing import List, Dict, Any, Optional


class NERService:
    """Biomedical Named Entity Recognition service"""
    
    def __init__(self, model_name: str = "en_core_sci_sm", confidence_threshold: float = 0.7):
        """
        Initialize NER service
        
        Args:
            model_name: SciSpacy model name
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nlp: Optional[object] = None
        self.hf_pipeline: Optional[object] = None
        self.backend: str = "none"  # one of: "spacy", "hf", "none"
        self._load_model()
    
    def _load_model(self):
        """Load the configured spaCy/SciSpaCy NER model or fall back to Hugging Face pipeline."""
        # 1) Try spaCy / SciSpaCy model
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            self.backend = "spacy"
            return
        except Exception as e:
            print(f"Warning: Could not load spaCy/SciSpaCy model '{self.model_name}': {e}")
            if str(self.model_name).startswith("en_core_sci_"):
                print(
                    "Hint: SciSpaCy models require installing the model package, e.g.\n"
                    "  pip install https://github.com/allenai/scispacy/releases/download/v0.5.4/"
                    "en_core_sci_sm-0.5.4.tar.gz\n"
                    "And SciSpaCy currently targets Python <=3.11 due to SciPy constraints."
                )
            else:
                print(
                    "Hint: For standard spaCy, install a model, e.g. 'python -m spacy download en_core_web_sm', "
                    "and set model_name to 'en_core_web_sm'."
                )

        # 2) Try Hugging Face token-classification pipeline
        try:
            from transformers import pipeline

            hf_model_name = self.model_name
            # Allow using prefix 'hf:' to indicate an HF model explicitly
            if hf_model_name.startswith("hf:"):
                hf_model_name = hf_model_name[3:]

            # If model looked like a spaCy package name, default to a biomedical HF model instead
            if hf_model_name.startswith("en_core_"):
                hf_model_name = "d4data/biomedical-ner-all"

            self.hf_pipeline = pipeline(
                "token-classification",
                model=hf_model_name,
                aggregation_strategy="simple",
            )
            self.backend = "hf"
            print(f"Loaded Hugging Face NER model: {hf_model_name}")
            return
        except Exception as e:
            print(f"Warning: Could not initialize Hugging Face NER pipeline: {e}")
            print("Falling back to placeholder NER (no entities will be extracted).")
            self.backend = "none"
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomedical entities from text
        
        Args:
            text: Input text
        
        Returns:
            List of entities with text, type, start, end, confidence
        """
        if self.backend == "none":
            # Placeholder: return empty list if model not loaded
            return []
        
        entities: List[Dict[str, Any]] = []

        if self.backend == "spacy":
            doc = self.nlp(text)
            for ent in getattr(doc, "ents", []):
                entity = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,  # spaCy/SciSpaCy typically don't provide probabilities
                }
                if entity["confidence"] >= self.confidence_threshold:
                    entities.append(entity)
            return entities

        if self.backend == "hf" and self.hf_pipeline is not None:
            results = self.hf_pipeline(text)
            for r in results:
                # transformers pipeline with aggregation_strategy="simple" returns entity_group
                start = int(r.get("start", 0))
                end = int(r.get("end", start))
                score = float(r.get("score", 1.0))
                entity = {
                    "text": r.get("word", text[start:end]),
                    "type": r.get("entity_group", r.get("entity", "ENTITY")),
                    "start": start,
                    "end": end,
                    "confidence": score,
                }
                if entity["confidence"] >= self.confidence_threshold:
                    entities.append(entity)
            return entities

        return []
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract medical terms from text (simplified version)
        
        Returns:
            List of medical term strings
        """
        entities = self.extract_entities(text)
        return [ent["text"] for ent in entities]
