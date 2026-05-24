#!/usr/bin/env python3
"""
build_faiss_index.py — Build FAISS index from pre-computed embeddings.npy

Reads embeddings produced by encode_documents.py and saves a FAISS index.
Also saves a doc_ids.json manifest so the retriever can map vector positions
back to PMIDs at query time.

Paths default to config data.embeddings_path / faiss.save_path when --config
is provided; explicit flags override.

Performance: uses all available CPU threads for FAISS ops; optionally builds
on GPU then converts to CPU index for a portable save.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import faiss
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_faiss_index(embeddings_path: Path, n_vectors: int, dim: int,
                      index_type: str = "IndexFlatIP", use_gpu: bool = False,
                      chunk_size: int = 100_000) -> faiss.Index:
    """Add embeddings to a FAISS index chunk-by-chunk to stay within RAM budget."""
    if index_type == "IndexFlatIP":
        cpu_index = faiss.IndexFlatIP(dim)
    elif index_type == "IndexFlatL2":
        cpu_index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    use_gpu = use_gpu and faiss.get_num_gpus() > 0
    if use_gpu:
        print(f"[INFO] Building on {faiss.get_num_gpus()} GPU(s), will convert to CPU index")
        target = faiss.index_cpu_to_all_gpus(cpu_index)
    else:
        target = cpu_index

    # Reload mmap inside this function to get a clean float32 view
    mmap = np.load(str(embeddings_path), mmap_mode="r")
    normalize = index_type == "IndexFlatIP"

    for start in range(0, n_vectors, chunk_size):
        end = min(start + chunk_size, n_vectors)
        chunk = mmap[start:end].astype(np.float32)  # one chunk in RAM at a time
        if normalize:
            faiss.normalize_L2(chunk)               # in-place, no extra alloc
        target.add(chunk)
        if start % (chunk_size * 10) == 0:
            print(f"  [{end:,}/{n_vectors:,}] vectors added", flush=True)

    if use_gpu:
        return faiss.index_gpu_to_cpu(target)
    return cpu_index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings.npy")
    parser.add_argument("--config", default="configs/fullpipeline.yaml",
                        help="Pipeline YAML config (default: configs/fullpipeline.yaml)")
    parser.add_argument("--embeddings",
                        help="Path to embeddings.npy (default: config data.embeddings_path)")
    parser.add_argument("--output",
                        help="Output path for FAISS index (default: config faiss.save_path)")
    parser.add_argument("--doc-ids",
                        help="Path to doc_ids JSON produced by encode_documents.py "
                             "(default: <embeddings_dir>/doc_ids.json)")
    parser.add_argument("--index-type", default=None,
                        help="FAISS index type (default: config faiss.index_type or IndexFlatIP)")
    parser.add_argument("--gpu", action="store_true",
                        help="Build index on GPU(s) then convert to CPU for saving "
                             "(faster add() for large corpora)")
    args = parser.parse_args()

    # Use all available CPU threads for FAISS parallel ops
    n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    faiss.omp_set_num_threads(n_threads)
    print(f"[INFO] FAISS using {n_threads} CPU threads")

    # Load config for defaults
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    elif args.config != "configs/fullpipeline.yaml":
        print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    data_cfg  = config.get("data", {})
    faiss_cfg = config.get("faiss", {})

    embeddings_path = Path(args.embeddings or data_cfg.get("embeddings_path", ""))
    output_path     = Path(args.output     or faiss_cfg.get("save_path", "faiss.index"))
    index_type      = args.index_type      or faiss_cfg.get("index_type", "IndexFlatIP")

    if not embeddings_path or not embeddings_path.exists():
        print(f"[ERROR] Embeddings not found: {embeddings_path}", file=sys.stderr)
        print("        Run encode_documents.py first.", file=sys.stderr)
        sys.exit(1)

    # Resolve doc_ids file: same directory as embeddings, named doc_ids.json
    doc_ids_path = Path(args.doc_ids) if args.doc_ids else embeddings_path.parent / "doc_ids.json"

    print(f"[INFO] Embeddings:  {embeddings_path}")
    print(f"[INFO] Output:      {output_path}")
    print(f"[INFO] Index type:  {index_type}")
    print(f"[INFO] GPU build:   {args.gpu}")
    print(f"[INFO] Doc-ids:     {doc_ids_path}")

    # Probe shape without loading into RAM
    print("Probing embeddings (memmap)...", flush=True)
    _probe = np.load(str(embeddings_path), mmap_mode="r")
    n_vectors, dim = _probe.shape
    on_disk_dtype = _probe.dtype
    del _probe
    print(f"[OK] {n_vectors:,} vectors of dim {dim} (on-disk dtype: {on_disk_dtype}, loaded as float32)")

    # Build index, normalising and adding chunk-by-chunk — peak RAM ≈ one chunk
    print(f"Building {index_type} index (chunked, GPU={args.gpu})...", flush=True)
    index = build_faiss_index(
        embeddings_path, n_vectors, dim, index_type, use_gpu=args.gpu
    )
    print(f"[OK] Index built: {index.ntotal:,} vectors")

    # Save index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"[OK] Saved FAISS index to {output_path}")

    # Save doc_ids manifest so retriever can map positions → PMIDs
    if doc_ids_path.exists():
        print(f"[INFO] doc_ids.json already exists at {doc_ids_path}, skipping write")
    else:
        print(f"[WARN] No doc_ids.json found at {doc_ids_path}")
        print("       The retriever needs this file to map FAISS positions to PMIDs.")
        print("       It is produced by encode_documents.py alongside embeddings.npy.")


if __name__ == "__main__":
    main()

