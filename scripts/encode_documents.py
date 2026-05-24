#!/usr/bin/env python3
"""
encode_documents.py — Batch-encode documents using configured encoder (GPU-ready)
Reads docs.jsonl and generates embeddings.npy and embeddings_manifest.json
"""

import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

# Optional: use actual encoders when available
try:
    from src.encoder.medcpt_encoder import MedCPTEncoder
    from src.encoder.biobert_encoder import BioBERTEncoder
except Exception:
    MedCPTEncoder = None
    BioBERTEncoder = None


def get_git_sha():
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


def compute_file_sha256(filepath):
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_lines(docs_path: str) -> int:
    """Count non-empty lines in a JSONL file.

    Uses the fast path: wc -l (kernel-buffered read) followed by a single
    pass to subtract blank lines.  On NFS this avoids a full Python-level
    23 GB read that can take 10-20 minutes before encoding even starts.
    Falls back to pure-Python counting if wc is unavailable.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["wc", "-l", docs_path],
            capture_output=True, text=True, check=True
        )
        total = int(result.stdout.split()[0])
        # wc -l counts newlines; subtract lines that are blank
        blank = subprocess.run(
            ["grep", "-c", "^$", docs_path],
            capture_output=True, text=True
        )
        blanks = int(blank.stdout.strip()) if blank.returncode == 0 else 0
        return total - blanks
    except Exception:
        n = 0
        with open(docs_path, "r") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

def stream_documents(docs_path):
    """Yield documents from JSONL file one by one"""
    with open(docs_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_encoder(config):
    enc_cfg = config.get("encoder", {})
    backend = enc_cfg.get("backend", "medcpt").lower()
    model_name = enc_cfg.get("model", "ncbi/MedCPT-Query-Encoder")
    article_model_name = enc_cfg.get("article_model", "ncbi/MedCPT-Article-Encoder")
    device = enc_cfg.get("device", "auto")
    if backend == "biobert" and BioBERTEncoder is not None:
        return BioBERTEncoder(model_name=model_name, device=device)
    if MedCPTEncoder is not None:
        return MedCPTEncoder(model_name=model_name, article_model_name=article_model_name, device=device)
    return None

def batch_encode_to_memmap(docs_path: str, config: dict, output_path: Path) -> int:
    """Encode documents in batches directly into a .npy memmap file.
    Returns the number of documents processed.
    """
    embedding_dim = int(config["encoder"]["embedding_dim"])
    batch_size = int(config["encoder"].get("batch_size", 32))
    total = count_lines(docs_path)
    if total == 0:
        # Create empty file
        mm = open_memmap(str(output_path), mode="w+", dtype=np.float16, shape=(0, embedding_dim))
        del mm
        return 0

    encoder = build_encoder(config)
    # float16: 61 GB for 40 M × 768-dim vs 122 GB for float32.
    # build_faiss_index.py loads chunks with .astype(np.float32) so the index
    # is still full-precision; query-time cosine similarity is unaffected.
    mm = open_memmap(str(output_path), mode="w+", dtype=np.float16, shape=(total, embedding_dim))

    def _encode_texts(texts: list[str]) -> np.ndarray:
        if encoder is None:
            # Placeholder: random embeddings
            emb = np.random.randn(len(texts), embedding_dim).astype(np.float32)
        else:
            emb = encoder.encode(texts, batch_size=batch_size, normalize=True)
        # Ensure float32 and normalized (defensive)
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-8)
        return emb

    i = 0
    buf_ids = []
    buf_texts = []
    all_doc_ids = []
    for doc in tqdm(stream_documents(docs_path), total=total, desc="Encoding"):
        # Build [title, abstract] pair — MedCPT-Article-Encoder expects sentence-pair input:
        #   tokenizer([title, abstract]) → [CLS] title [SEP] abstract [SEP]
        # Using both fields gives significantly better retrieval than abstract-only.
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or doc.get("text") or "").strip()
        doc_id = doc.get("doc_id") or doc.get("pmid")
        buf_ids.append(doc_id)
        # Pass as [title, abstract] pair; encoder.encode() handles pair tokenization
        buf_texts.append([title, abstract])
        all_doc_ids.append(doc_id)

        # Flush batch
        if len(buf_texts) >= batch_size:
            emb = _encode_texts(buf_texts)
            mm[i:i+len(emb), :] = emb
            i += len(emb)
            buf_ids.clear()
            buf_texts.clear()

    # Flush tail
    if buf_texts:
        emb = _encode_texts(buf_texts)
        mm[i:i+len(emb), :] = emb
        i += len(emb)

    del mm  # flush to disk

    # Save doc_ids so FAISS positions map back to PMIDs
    doc_ids_path = output_path.parent / "doc_ids.json"
    with open(doc_ids_path, "w") as f:
        json.dump(all_doc_ids, f)

    return i


def main():
    parser = argparse.ArgumentParser(description="Encode documents for RAG pipeline")
    parser.add_argument("--config", default="configs/fullpipeline.yaml",
                        help="Path to config YAML (default: configs/fullpipeline.yaml)")
    parser.add_argument("--output-dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Encoding batch size (overrides config; use 128-256 on A100)")
    parser.add_argument("--tmp-dir", default=None,
                        help="Fast local scratch dir for intermediate write (e.g. $TMPDIR). "
                             "Embeddings are written here first, then moved to --output-dir.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI batch-size overrides config
    if args.batch_size is not None:
        config.setdefault("encoder", {})["batch_size"] = args.batch_size

    # Source docs path
    docs_path = config["data"]["docs_path"]
    print(f"Preparing to encode documents from {docs_path}...")

    # Determine write target: fast local scratch if provided, else final output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tmp_dir:
        write_dir = Path(args.tmp_dir)
        write_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing intermediates to fast scratch: {write_dir}")
    else:
        write_dir = output_dir

    embeddings_path = write_dir / "embeddings.npy"
    article_model = config['encoder'].get('article_model', config['encoder']['model'])
    print(f"Encoding with article_encoder={article_model} "
          f"(query_encoder={config['encoder']['model']}, "
          f"device={config['encoder'].get('device','auto')}, "
          f"batch_size={config['encoder'].get('batch_size', 32)}, "
          f"input=title+abstract pair)...")
    n_docs = batch_encode_to_memmap(docs_path, config, embeddings_path)
    print(f"Encoded {n_docs} documents")

    # Move from scratch to final output dir if needed
    if args.tmp_dir:
        import shutil
        for fname in ("embeddings.npy", "doc_ids.json"):
            src = write_dir / fname
            dst = output_dir / fname
            if src.exists():
                print(f"Moving {src} → {dst}")
                shutil.move(str(src), str(dst))
        embeddings_path = output_dir / "embeddings.npy"

    print(f"Saved embeddings to {output_dir / 'embeddings.npy'} (n_docs={n_docs})")
    print(f"Saved doc_ids to   {output_dir / 'doc_ids.json'}")

    # Create manifest
    manifest = {
        "git_sha": get_git_sha(),
        "encoder": config["encoder"]["model"],
        "article_encoder": config["encoder"].get("article_model", ""),
        "embedding_dim": config["encoder"]["embedding_dim"],
        "batch_size": config["encoder"].get("batch_size", 32),
        "num_documents": n_docs,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_sha256": compute_file_sha256(docs_path),
        "embeddings_sha256": compute_file_sha256(str(output_dir / "embeddings.npy"))
    }

    manifest_path = output_dir / "embeddings_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
