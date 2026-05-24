"""
Utility functions for the RAG pipeline
"""

import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def get_git_sha() -> str:
    """Get current git commit SHA"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_run_manifest(
    run_id: str,
    config: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save run manifest for reproducibility
    """
    manifest = {
        "run_id": run_id,
        "git_sha": get_git_sha(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": config.get("pipeline", {}).get("seed", 42),
        "models": {
            "encoder": config.get("encoder", {}).get("model", "unknown"),
            "reranker": config.get("reranker", {}).get("model", "unknown"),
            "llm": config.get("llm", {}).get("model", "unknown")
        },
        "config": config
    }
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to JSONL file"""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_citation(doc_id: str, title: str, pub_date: str = None) -> str:
    """Format a citation for a document"""
    citation = f"[{doc_id}] {title}"
    if pub_date:
        citation += f" ({pub_date})"
    return citation
