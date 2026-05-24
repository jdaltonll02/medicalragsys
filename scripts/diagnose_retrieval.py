#!/usr/bin/env python3
"""
Diagnostic script: checks dense retrieval quality for a sample of BioASQ questions.

Outputs:
 - FAISS score distribution for top-k results
 - Whether golden PMIDs appear in top-500 FAISS results
 - BM25 score distribution and golden PMID presence
 - Inter-doc similarities between golden and submitted docs

Run on SLURM node (needs GPU/transformers + faiss):
    python scripts/diagnose_retrieval.py --config configs/fullpipeline.yaml \
           --golden golden_round3_testset.json --submission results/submission.json \
           --n-questions 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_faiss(config):
    import faiss
    faiss_path = config["faiss"]["save_path"]
    doc_ids_path = Path(faiss_path).parent / "doc_ids.json"
    index = faiss.read_index(faiss_path)
    with open(doc_ids_path) as f:
        doc_ids = json.load(f)
    pmid_to_row = {str(p): i for i, p in enumerate(doc_ids)}
    print(f"[FAISS] {index.ntotal:,} vectors loaded")
    return index, doc_ids, pmid_to_row


def load_encoder(config):
    from src.encoder.medcpt_encoder import MedCPTEncoder
    enc_cfg = config.get("encoder", {})
    enc = MedCPTEncoder(
        model_name=enc_cfg.get("model", "ncbi/MedCPT-Query-Encoder"),
        article_model_name=enc_cfg.get("article_model", "ncbi/MedCPT-Article-Encoder"),
        device="auto"
    )
    return enc


def diagnose_question(q_id, q_body, golden_pmids, submitted_pmids,
                      faiss_index, doc_ids, pmid_to_row, encoder,
                      bm25=None, top_k=500):
    print(f"\n{'='*70}")
    print(f"Question: {q_body[:100]}")
    print(f"Golden PMIDs ({len(golden_pmids)}): {golden_pmids[:5]}")
    print(f"Submitted PMIDs (top-5): {submitted_pmids[:5]}")

    # ── 1. Encode query ──────────────────────────────────────────────────
    q_emb = encoder.encode_query(q_body)
    print(f"\n[Query embedding] norm={np.linalg.norm(q_emb):.4f}")

    # ── 2. FAISS search ──────────────────────────────────────────────────
    scores, indices = faiss_index.search(q_emb.reshape(1, -1).astype("float32"), top_k)
    scores, indices = scores[0], indices[0]

    print(f"\n[FAISS top-{top_k}]")
    print(f"  Score range:  min={scores.min():.4f}  max={scores.max():.4f}  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")

    # Check which golden PMIDs appear and at what rank
    golden_set = set(str(p) for p in golden_pmids)
    submitted_set = set(str(p) for p in submitted_pmids)
    golden_ranks = {}
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        pmid = str(doc_ids[int(idx)])
        if pmid in golden_set:
            golden_ranks[pmid] = (rank, float(score))

    print(f"  Golden PMIDs found in top-{top_k}: {len(golden_ranks)}/{len(golden_set)}")
    for pmid, (rank, score) in sorted(golden_ranks.items(), key=lambda x: x[1][0]):
        print(f"    PMID {pmid}: rank={rank}, score={score:.4f}")
    if not golden_ranks:
        print(f"    *** NONE of the {len(golden_set)} golden PMIDs in top-{top_k} ***")

    # Check where submitted PMIDs rank
    submitted_ranks = {}
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        pmid = str(doc_ids[int(idx)])
        if pmid in submitted_set:
            submitted_ranks[pmid] = (rank, float(score))
    print(f"  Submitted PMIDs found in top-{top_k}: {len(submitted_ranks)}/{len(submitted_set)}")
    for pmid, (rank, score) in sorted(submitted_ranks.items(), key=lambda x: x[1][0])[:5]:
        print(f"    PMID {pmid}: rank={rank}, score={score:.4f}")

    # Top-10 FAISS results
    print(f"\n  Top-10 FAISS results:")
    for rank in range(min(10, len(indices))):
        pmid = str(doc_ids[int(indices[rank])])
        flag = " ← GOLDEN" if pmid in golden_set else (" ← SUBMITTED" if pmid in submitted_set else "")
        print(f"    {rank+1:2d}. PMID {pmid}  score={scores[rank]:.4f}{flag}")

    # ── 3. BM25 search ──────────────────────────────────────────────────
    if bm25 is not None:
        try:
            bm25_results = bm25.search(q_body, top_k=100)
            bm25_pmids = [r["doc_id"] for r in bm25_results]
            bm25_scores = [r["score"] for r in bm25_results]
            bm25_golden = [p for p in bm25_pmids if str(p) in golden_set]
            print(f"\n[BM25 top-100]")
            print(f"  Score range: min={min(bm25_scores):.2f}  max={max(bm25_scores):.2f}")
            print(f"  Golden PMIDs found: {len(bm25_golden)}/{len(golden_set)}")
            print(f"  Top-5 BM25 PMIDs: {bm25_pmids[:5]}")
        except Exception as e:
            print(f"\n[BM25] Error: {e}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fullpipeline.yaml")
    parser.add_argument("--golden", default="golden_round3_testset.json")
    parser.add_argument("--submission", default="results/submission.json")
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading FAISS index...")
    faiss_index, doc_ids, pmid_to_row = load_faiss(config)

    print("Loading encoder...")
    encoder = load_encoder(config)

    # Load BM25 if ES is up
    bm25 = None
    try:
        from src.retrieval.bm25_retriever import BM25Retriever
        bm25_cfg = config.get("bm25", {})
        bm25 = BM25Retriever(
            host=bm25_cfg.get("elasticsearch_host", "localhost"),
            port=bm25_cfg.get("elasticsearch_port", 9200),
            index_name=bm25_cfg.get("index_name", "medical_docs_3"),
        )
        bm25.es.info()
        print("BM25 (ES) connected")
    except Exception as e:
        print(f"BM25 unavailable: {e} — FAISS-only diagnostic")

    with open(args.golden) as f:
        golden = {q["id"]: q for q in json.load(f)["questions"]}
    with open(args.submission) as f:
        sub = {q["id"]: q for q in json.load(f)["questions"]}

    # Pick questions with smallest golden set (easiest to find)
    sample = sorted(
        [(q["id"], q) for q in golden.values() if q["id"] in sub],
        key=lambda x: len(x[1]["documents"])
    )[:args.n_questions]

    for q_id, gq in sample:
        sq = sub[q_id]
        diagnose_question(
            q_id=q_id,
            q_body=gq["body"],
            golden_pmids=gq["documents"],
            submitted_pmids=sq["documents"],
            faiss_index=faiss_index,
            doc_ids=doc_ids,
            pmid_to_row=pmid_to_row,
            encoder=encoder,
            bm25=bm25,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
