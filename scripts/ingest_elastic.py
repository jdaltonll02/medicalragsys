#!/usr/bin/env python3
"""
ingest_elastic.py
-----------------
Ingest full PubMed corpus into Elasticsearch for BM25 retrieval.

• Streams JSONL — OOM safe for multi-million-doc corpora
• Disables ES refresh during bulk ingest, restores on exit
• tqdm progress bar with running success/failure counts
• Corpus path defaults to config data.docs_path
"""

import argparse
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_index(es, index_name, force=False):
    if es.indices.exists(index=index_name):
        if not force:
            raise RuntimeError(
                f"Index '{index_name}' already exists. "
                f"Use --force to recreate it."
            )
        print(f"[WARN] Deleting existing index '{index_name}'")
        es.indices.delete(index=index_name)

    body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",   # disabled for bulk ingest, restored after
            },
            "analysis": {
                "analyzer": {
                    "english": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id":        {"type": "keyword"},
                "pmid":          {"type": "keyword"},
                "title":         {"type": "text", "analyzer": "english"},
                "abstract":      {"type": "text", "analyzer": "english"},
                "pub_date": {
                    "type": "date",
                    "format": (
                        "yyyy||yyyy-MM||yyyy-MM-dd||"
                        "MMM yyyy||strict_date_optional_time"
                    )
                },
                "ingested_at":   {"type": "date"},
                "source":        {"type": "keyword"}
            }
        }
    }

    es.indices.create(index=index_name, body=body)
    print(f"[OK] Created index '{index_name}'")
    # Wait for primary shard to become active before bulk ingest
    es.cluster.health(
        index=index_name,
        wait_for_active_shards="1",
        wait_for_status="yellow",
        timeout="120s",
    )
    print(f"[OK] Primary shard active")


def generate_actions(docs_path, index_name):
    ingest_time = datetime.utcnow().isoformat()
    with open(docs_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                continue

            doc_id = doc.get("doc_id") or doc.get("pmid")
            if not doc_id:
                print(f"[WARN] Line {line_num}: Missing doc_id/pmid, skipping", file=sys.stderr)
                continue

            yield {
                "_index": index_name,
                "_id": str(doc_id),
                "_source": {
                    "doc_id":      str(doc_id),
                    "pmid":        str(doc_id),
                    "title":       doc.get("title", ""),
                    "abstract":    doc.get("abstract", ""),
                    "pub_date":    doc.get("pub_date"),
                    "ingested_at": ingest_time,
                    "source":      "pubmed"
                }
            }


def ingest_corpus(es, docs_path, index_name, chunk_size):
    print(f"[INFO] Streaming ingest from {docs_path} ...")
    success = 0
    failed = 0

    pbar = tqdm(
        streaming_bulk(
            es,
            generate_actions(docs_path, index_name),
            chunk_size=chunk_size,
            raise_on_error=False,
            max_retries=3,
        ),
        desc="Ingesting",
        unit=" docs",
        unit_scale=True,
        mininterval=2.0,
    )

    for ok, result in pbar:
        if ok:
            success += 1
        else:
            failed += 1
            if failed <= 10:
                print(f"[WARN] Ingest error: {result}", file=sys.stderr)
        if (success + failed) % 50_000 == 0:
            pbar.set_description(f"Ingesting ({success:,} ok, {failed} failed)")

    pbar.close()
    return success, failed


def restore_index_settings(es, index_name):
    try:
        es.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "1s", "number_of_replicas": 0}}
        )
        es.indices.refresh(index=index_name)
        print("[OK] ES refresh restored and index refreshed")
    except Exception as e:
        print(f"[WARN] Could not restore ES settings: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Ingest full PubMed corpus into Elasticsearch")
    parser.add_argument("--config", default="configs/fullpipeline.yaml",
                        help="Pipeline YAML config (default: configs/fullpipeline.yaml)")
    parser.add_argument("--docs",
                        help="PubMed JSONL corpus (default: config data.docs_path)")
    parser.add_argument("--index", help="Index name (overrides config)")
    parser.add_argument("--host",  help="Elasticsearch host (overrides config)")
    parser.add_argument("--port",  type=int, help="Elasticsearch port (overrides config)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Bulk request chunk size (default 1000)")
    parser.add_argument("--force", action="store_true",
                        help="Recreate index if it already exists")
    args = parser.parse_args()

    config = load_config(args.config)
    bm25_cfg = config.get("bm25", {})

    host       = args.host  or bm25_cfg.get("elasticsearch_host", "localhost")
    port       = args.port  or bm25_cfg.get("elasticsearch_port", 9200)
    index_name = args.index or bm25_cfg.get("index_name", "medical_docs")
    docs_path  = args.docs  or config.get("data", {}).get("docs_path")

    if not docs_path:
        print("[ERROR] No corpus path specified (--docs or config data.docs_path)", file=sys.stderr)
        sys.exit(1)
    if not Path(docs_path).exists():
        print(f"[ERROR] Corpus not found: {docs_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Host:       {host}:{port}")
    print(f"[INFO] Index:      {index_name}")
    print(f"[INFO] Corpus:     {docs_path}")
    print(f"[INFO] Chunk size: {args.chunk_size}")

    es = Elasticsearch(
        [f"http://{host}:{port}"],
        request_timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )
    if not es.ping():
        print(f"[ERROR] Cannot reach Elasticsearch at {host}:{port}", file=sys.stderr)
        sys.exit(1)
    print("[OK] Connected to Elasticsearch")

    create_index(es, index_name, force=args.force)

    try:
        success, failed = ingest_corpus(
            es, docs_path, index_name, args.chunk_size
        )
    finally:
        restore_index_settings(es, index_name)

    print(f"\n[DONE] Ingested {success:,} documents ({failed} failures)")

    try:
        stats = es.indices.stats(index=index_name)
        doc_count  = stats["indices"][index_name]["primaries"]["docs"]["count"]
        store_bytes = stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"]
        print(f"[INFO] Index docs:  {doc_count:,}")
        print(f"[INFO] Index size:  {store_bytes / 1024**3:.2f} GB")
    except Exception:
        pass

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
