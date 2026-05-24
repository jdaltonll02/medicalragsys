# ### Data Processing Notebook

import os
from pathlib import Path
import gzip
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import faiss
import numpy as np
from multiprocessing import Pool

# -------------------------------------------------------------------
# Paths and setup
# -------------------------------------------------------------------

BASELINE_DIR = Path("/data/user_data/jgibson2/bioask_pubmed_dataset/baseline")
UPDATE_DIR   = Path("/data/user_data/jgibson2/bioask_pubmed_dataset/updates")
JSON_DIR     = Path("/data/user_data/jgibson2/bioask_pubmed_dataset/parsed")

JSON_DIR.mkdir(parents=True, exist_ok=True)

baseline_files = list(BASELINE_DIR.glob("*.xml.gz"))
update_files   = list(UPDATE_DIR.glob("*.xml.gz"))

all_files = baseline_files + update_files

print("Baseline files:", len(baseline_files))
print("Update files:", len(update_files))
print("Total XML files:", len(all_files))


# -------------------------------------------------------------------
# Snapshot Window
# -------------------------------------------------------------------

baseline_end = datetime(2025, 1, 8, 23, 10)
snapshot_end = datetime(2026, 1, 13, 14, 2)


# -------------------------------------------------------------------
# XML → JSON Parser
# -------------------------------------------------------------------

def parse_pubmed(xml_gz_path):
    """
    Parse a PubMed XML.GZ file and yield article dictionaries.
    """
    with gzip.open(xml_gz_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            pmid = elem.findtext("./MedlineCitation/PMID")
            title = elem.findtext(".//ArticleTitle") or ""
            abstract = " ".join(
                t.text or "" for t in elem.findall(".//AbstractText")
            )

            if pmid and (title or abstract):
                yield {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract
                }

            elem.clear()


# -------------------------------------------------------------------
# Process all files
# -------------------------------------------------------------------

OUT_FILE = JSON_DIR / "pubmed_corpus.jsonl"

def process_one_file(xml_file):
    count = 0
    lines = []
    for article in parse_pubmed(xml_file):
        lines.append(json.dumps(article))
        count += 1
    return count, lines


if __name__ == "__main__":
    with Pool(processes=8) as pool, open(OUT_FILE, "a") as out_f:
        for xml_file, (count, lines) in zip(
            all_files, pool.imap(process_one_file, all_files)
        ):
            for line in lines:
                out_f.write(line + "\n")
            print(f"Processed {xml_file.name}: {count} articles")

    # -------------------------------------------------------------------
    # Sanity check: read first article
    # -------------------------------------------------------------------

    with open(OUT_FILE, "r") as f:
        first_line = f.readline()
        first_article = json.loads(first_line)

    print(first_article)

