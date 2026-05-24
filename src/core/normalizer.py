"""
Text normalization utilities for medical queries and documents
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text by:
    - Converting to lowercase
    - Removing extra whitespace
    - Normalizing unicode characters
    - Removing special characters (preserving medical terms)
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_medical_query(
    query: str,
    lowercase: bool = True,
    remove_punct: bool = True,
) -> str:
    """
    Normalize medical query with special handling for medical terms.
    Preserves hyphens in terms like COVID-19 and IL-6.

    Args:
        query:       Raw query string.
        lowercase:   If False, case is preserved (useful when the dense
                     encoder is case-sensitive, e.g. MedCPT).
        remove_punct: If False, punctuation is left in place after
                     whitespace normalization.
    """
    if not query:
        return ""

    # Unicode normalisation is always applied — only affects ligatures /
    # accented characters, never changes case or punctuation.
    query = unicodedata.normalize("NFKD", query)

    if lowercase:
        query = query.lower()

    # Collapse whitespace
    query = re.sub(r'\s+', ' ', query).strip()

    if remove_punct:
        query = remove_punctuation(query, preserve_medical=True)

    return query


def remove_punctuation(text: str, preserve_medical: bool = True) -> str:
    """
    Remove punctuation while optionally preserving medical terms
    """
    if preserve_medical:
        # Keep hyphens in medical terms like COVID-19
        text = re.sub(r'[^\w\s\-]', '', text)
    else:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


def truncate_text(text: str, max_length: int = 512, suffix: str = "...") -> str:
    """
    Truncate text to maximum length, adding suffix if truncated
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
