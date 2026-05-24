#!/usr/bin/env python3
"""
Run BioASQ RAG pipeline against the full pre-built corpus indices.

Assumes:
  - FAISS index already built by build_faiss_index.py
  - Elasticsearch already ingested by ingest_elastic.py (or FAISS-only in config)
  - doc_ids.json produced by encode_documents.py

Pass --testset to any BioASQ-format JSON file.
--skip-indexing (default) loads pre-built FAISS; omit only to re-encode from scratch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from tqdm import tqdm

from src.pipeline.med_rag import MedicalRAGPipeline
from src.core.synergy_formatter import SnippetExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("bioasq-pipeline")


# ---------------------------------------------------------------------
# Keyword filter: prefer docs containing query terms
# ---------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "or", "and", "that", "this",
    "it", "its", "not", "no", "but", "if", "what", "which", "who", "how",
    "when", "where", "why", "their", "they", "them", "we", "our", "your",
    "after", "about", "between", "through", "during", "before", "than",
}

def _keyword_filter(
    query: str,
    docs: List[Dict[str, Any]],
    min_passing: int = 5,
) -> List[Dict[str, Any]]:
    """Return docs reordered so keyword-matching docs come first."""
    terms = [
        w.lower().strip(".,?!()[]") for w in query.split()
        if len(w) > 3 and w.lower() not in _STOPWORDS
    ]
    if not terms:
        return docs

    passing, failing = [], []
    for doc in docs:
        text = ((doc.get("title") or "") + " " + (doc.get("abstract") or "")).lower()
        if any(t in text for t in terms):
            passing.append(doc)
        else:
            failing.append(doc)

    # Always return at least min_passing docs to avoid starving the answer
    if len(passing) < min_passing:
        passing = passing + failing[: min_passing - len(passing)]
        failing = failing[max(0, min_passing - len(passing)):]

    return passing + failing


# ---------------------------------------------------------------------
# Load BioASQ testset
# ---------------------------------------------------------------------
def load_bioasq_testset(testset_path: Path) -> List[Dict[str, Any]]:
    logger.info(f"Loading BioASQ testset from {testset_path}")
    with open(testset_path, "r") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} BioASQ questions")
    return questions


# ---------------------------------------------------------------------
# Stream PubMed corpus (JSONL) — used only when not skip-indexing
# ---------------------------------------------------------------------
def stream_pubmed_corpus(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    logger.info(f"Streaming PubMed corpus from {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            pmid = doc.get("doc_id") or doc.get("pmid")
            abstract = doc.get("abstract", "")
            if not pmid or not abstract:
                continue
            title = doc.get("title", "")
            yield {
                "doc_id": str(pmid),
                "title": title,
                "abstract": abstract,
                "text": f"{title}\n\n{abstract}".strip(),
                "metadata": {"pmid": pmid}
            }


# ---------------------------------------------------------------------
# Corpus text lookup via byte-offset index (skip-indexing mode)
# Avoids loading the full 40M-doc corpus into RAM; seeks per query.
# Offset file is built once and reused across runs.
# ---------------------------------------------------------------------
class CorpusLookup:
    """Random-access title/abstract lookup by FAISS row index."""

    def __init__(self, docs_path: Path, offsets: np.ndarray):
        self._offsets = offsets
        self._f = open(docs_path, "rb")

    def get(self, row_index: int):
        """Return (title, abstract) for the given FAISS row, or ("", "")."""
        if row_index < 0 or row_index >= len(self._offsets):
            return "", ""
        offset = int(self._offsets[row_index])
        if offset < 0:
            return "", ""
        self._f.seek(offset)
        try:
            doc = json.loads(self._f.readline())
            return doc.get("title", ""), doc.get("abstract", "")
        except Exception:
            return "", ""

    def close(self):
        self._f.close()


def _build_corpus_offset_index(
    docs_path: Path,
    doc_ids: List[str],
    save_path: Path,
) -> np.ndarray:
    """Scan corpus once and record byte offset per FAISS row. Saved as .npy."""
    logger.info(f"Building corpus offset index (scanning {docs_path})...")
    pmid_to_row = {pmid: i for i, pmid in enumerate(doc_ids)}
    offsets = np.full(len(doc_ids), -1, dtype=np.int64)

    with open(docs_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue
            try:
                doc = json.loads(line_str)
                pmid = str(doc.get("doc_id") or doc.get("pmid", ""))
                row = pmid_to_row.get(pmid, -1)
                if row >= 0:
                    offsets[row] = pos
            except Exception:
                continue

    np.save(str(save_path), offsets)
    n_found = int((offsets >= 0).sum())
    logger.info(
        f"Offset index saved to {save_path} "
        f"({n_found:,}/{len(doc_ids):,} docs located)"
    )
    return offsets


def _load_or_build_offset_index(
    docs_path: Path,
    doc_ids: List[str],
    faiss_save_path: str,
) -> np.ndarray:
    save_path = Path(faiss_save_path).parent / "corpus_offsets.npy"
    if save_path.exists():
        logger.info(f"Loading corpus offset index from {save_path}")
        return np.load(str(save_path))
    return _build_corpus_offset_index(docs_path, doc_ids, save_path)


# ---------------------------------------------------------------------
# High-performance incremental indexing (used when NOT skip-indexing)
# ---------------------------------------------------------------------
def index_pubmed_stream(
    pipeline: MedicalRAGPipeline,
    corpus_stream: Iterator[Dict[str, Any]],
    batch_size: int = 5_000,
):
    import time

    fallback_cfg = pipeline.config.get("fallback", {})
    use_faiss_only = fallback_cfg.get("use_faiss_only", False) or \
                     not fallback_cfg.get("use_elasticsearch", True)

    bm25 = pipeline.bm25_retriever
    es = bm25.es if (bm25 is not None and not use_faiss_only) else None

    if es is not None:
        try:
            bm25.reset_index()
        except Exception:
            pass
        es.indices.put_settings(
            index=bm25.index_name,
            body={"index": {"refresh_interval": "-1", "number_of_replicas": 0}}
        )
    else:
        logger.info("Elasticsearch not used; skipping bulk-indexing tuning")

    logger.info("Indexing PubMed documents (streaming + bulk mode)")
    batch, total = [], 0

    for doc in tqdm(corpus_stream, desc="Indexing", unit=" docs",
                    unit_scale=True, mininterval=1.0, file=sys.stderr):
        batch.append(doc)
        if len(batch) >= batch_size:
            t0 = time.time()
            pipeline.index_documents(batch, reset_index=False, index_fallback=False)
            total += len(batch)
            logger.info(f"Batch done in {time.time()-t0:.1f}s — total: {total:,}")
            batch.clear()

    if batch:
        pipeline.index_documents(batch, reset_index=False, index_fallback=False)
        total += len(batch)

    logger.info(f"Finished indexing {total:,} documents")

    if es is not None:
        es.indices.put_settings(
            index=bm25.index_name,
            body={"index": {"refresh_interval": "1s", "number_of_replicas": 0}}
        )
        es.indices.refresh(index=bm25.index_name)


# ---------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------
def run_pipeline(
    questions: List[Dict[str, Any]],
    config_path: Path,
    skip_indexing: bool = True,
) -> List[Dict[str, Any]]:

    logger.info("Loading pipeline configuration")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pubmed_path = Path(config.get("data", {}).get("docs_path", ""))
    if not pubmed_path.exists():
        logger.error(f"Corpus not found: {pubmed_path}")
        sys.exit(1)

    pipeline = MedicalRAGPipeline(config)

    # -----------------------------------------------------------------
    # INDEXING
    # -----------------------------------------------------------------
    if skip_indexing:
        logger.info("--skip-indexing: using pre-built FAISS index")
        faiss_save_path = config.get("faiss", {}).get("save_path", "")
        doc_ids_path = Path(faiss_save_path).parent / "doc_ids.json" if faiss_save_path else None

        if doc_ids_path and doc_ids_path.exists():
            with open(doc_ids_path) as f:
                doc_ids = json.load(f)
            pipeline.faiss_index.set_doc_ids(doc_ids)
            logger.info(f"Loaded {len(doc_ids):,} doc_ids from {doc_ids_path}")

            # Build/load byte-offset index for corpus text lookup
            offsets = _load_or_build_offset_index(pubmed_path, doc_ids, faiss_save_path)
            pipeline._corpus_lookup = CorpusLookup(pubmed_path, offsets)
            logger.info("Corpus text lookup ready")
        else:
            logger.warning(f"doc_ids.json not found at {doc_ids_path}; FAISS results will lack text")
    else:
        corpus_stream = stream_pubmed_corpus(pubmed_path)
        index_pubmed_stream(pipeline, corpus_stream)

    # -----------------------------------------------------------------
    # RAG INFERENCE
    # -----------------------------------------------------------------
    logger.info("Running RAG inference")
    results = []
    corpus_lookup: CorpusLookup | None = getattr(pipeline, "_corpus_lookup", None)

    for idx, q in enumerate(questions, 1):
        query = q["body"]
        qid = q["id"]
        qtype = q.get("type")

        if idx % 10 == 0:
            logger.info(f"Processing question {idx}/{len(questions)}")

        output = pipeline.process_query(query, question_type=qtype)
        final_docs = output.get("final_documents", [])
        reranked_docs = output.get("reranked_documents", [])

        # Use the broader reranked pool as candidate set for filtering
        candidate_pool = reranked_docs if reranked_docs else final_docs

        # Enrich full candidate pool with title/abstract
        if corpus_lookup is not None:
            for doc in candidate_pool:
                if not doc.get("abstract"):
                    row = doc.get("index")
                    if isinstance(row, int):
                        doc["title"], doc["abstract"] = corpus_lookup.get(row)

        # Reorder so keyword-matching docs come first, then take top 10
        candidate_pool = _keyword_filter(query, candidate_pool)
        top_docs = candidate_pool[:10]

        snippets = SnippetExtractor.extract_snippets(
            query=query,
            documents=top_docs,
            max_snippets=10
        )

        results.append({
            "id": qid,
            "body": query,
            "type": qtype,
            "answer_ready": q.get("answerReady", False),
            "answer": output.get("answer"),
            "documents": [d["doc_id"] for d in top_docs],
            "snippets": snippets
        })

    if corpus_lookup is not None:
        corpus_lookup.close()

    return results


# ---------------------------------------------------------------------
# Save results (BioASQ submission format)
# ---------------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = {"questions": []}

    for r in results:
        answer = r.get("answer", "")
        qtype = r.get("type", "")

        question_entry = {
            "id": r["id"],
            "body": r.get("body", ""),
            "type": qtype,
            "documents": r["documents"],
            "snippets": r["snippets"]
        }

        if not answer:
            if qtype == "yesno":
                answer = "Unable to determine from available evidence"
            elif qtype == "factoid":
                answer = "Information not available"
            elif qtype == "list":
                answer = "No specific items identified"
            else:
                answer = "Insufficient evidence to provide a comprehensive answer"

        words = answer.split()
        ideal_answer = " ".join(words[:200]) if len(words) > 200 else answer

        if qtype == "yesno":
            answer_lower = answer.lower()
            if "yes" in answer_lower[:50] and "no" not in answer_lower[:50]:
                exact_ans = "yes"
            elif "no" in answer_lower[:50]:
                exact_ans = "no"
            else:
                exact_ans = "no" if any(
                    w in answer_lower for w in ["unable", "insufficient", "not available", "unknown"]
                ) else "yes"
            question_entry["exact_answer"] = exact_ans
            question_entry["ideal_answer"] = ideal_answer

        elif qtype == "factoid":
            # The LLM generates "Entity Name\n\nExplanation..."; take only the first line
            first_line = answer.split("\n")[0].strip()
            entities = []
            for sep in [",", ";", " and ", " or "]:
                if sep in first_line:
                    entities = [e.strip() for e in first_line.split(sep) if e.strip()]
                    break
            if not entities:
                entities = [first_line] if first_line else ["Information not available"]
            question_entry["exact_answer"] = [[e[:100]] for e in entities[:5] if e.strip()] \
                or [["Information not available"]]
            question_entry["ideal_answer"] = ideal_answer

        elif qtype == "list":
            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            items = []

            # First pass: numbered/bulleted lines
            for line in lines:
                if line and (line[0].isdigit() or line[0] in "-*•"):
                    item = line.lstrip("0123456789.-*• ").strip().rstrip(".,;:")
                    if item and len(items) < 100:
                        items.append([item[:100]])

            # Second pass: if no structured items, try comma-splitting prose lines
            if not items:
                _skip = ("the ", "these ", "following", "include", "such as", "e.g.", "i.e.", "there are", "here are")
                for line in lines:
                    if any(line.lower().startswith(p) for p in _skip):
                        continue
                    parts = [p.strip().rstrip(".,;:") for p in line.replace(";", ",").split(",")]
                    parts = [p for p in parts if p and len(p.split()) <= 8]
                    if len(parts) >= 2:
                        for p in parts:
                            if len(items) < 100:
                                items.append([p[:100]])

            question_entry["exact_answer"] = items[:100] or [["No specific items identified"]]
            question_entry["ideal_answer"] = ideal_answer

        else:  # summary
            question_entry["ideal_answer"] = ideal_answer

        submission["questions"].append(question_entry)

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    logger.info(f"Saved BioASQ submission to {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("BioASQ Full Corpus RAG Pipeline")
    parser.add_argument("--testset", type=Path, required=True,
                        help="BioASQ testset JSON (e.g. test_data/round_4/testset_round_4.json)")
    parser.add_argument("--config", type=Path, default=Path("configs/fullpipeline.yaml"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Output submission JSON file path")
    parser.add_argument("--skip-indexing", action="store_true", default=True,
                        help="Load pre-built FAISS index instead of re-encoding (default: true)")
    parser.add_argument("--no-skip-indexing", dest="skip_indexing", action="store_false",
                        help="Re-encode corpus from scratch (slow, rarely needed)")
    args = parser.parse_args()

    if not args.testset.exists():
        logger.error(f"Testset not found: {args.testset}")
        sys.exit(1)

    questions = load_bioasq_testset(args.testset)
    results = run_pipeline(
        questions=questions,
        config_path=args.config,
        skip_indexing=args.skip_indexing,
    )
    save_results(results, args.output)


if __name__ == "__main__":
    main()
