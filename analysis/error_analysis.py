"""
SynapFlow BioASQ Synergy — Submission vs Expert Feedback Error Analysis

Metrics computed:
  Document retrieval : Precision, Recall, F1, MRR (per question and aggregate)
  Snippet retrieval  : Precision, Recall, F1 (exact text match)
  Yes/No answers     : Accuracy
  Factoid answers    : Precision@1 (first submitted entity matches any golden)
  List answers       : Precision, Recall, F1 over entity sets
  Ideal answers      : ROUGE-1/2/L
  Cross-cutting      : per-type breakdowns, score distributions, error taxonomy

Outputs (all in analysis/):
  metrics.json                — full numeric results
  plots/                      — PNG visualisations
"""

import json
import re
import math
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
SUB_PATH   = ROOT / "analysis" / "v2026-3-SynapFlow.json"   # actual competition submission
FB_PATH    = ROOT / "results" / "round3_feedback.json"
GOLDEN_PATH = ROOT / "test_data" / "round_3" / "golden_round3_testset.json"
OUT_DIR    = ROOT / "analysis"
PLOT_DIR   = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── malformed answer patterns ──────────────────────────────────────────────────
MALFORMED_PATTERNS = [
    r"information not available",
    r"no specific items? identified",
    r"not available",
    r"the main entity is",
    r"cannot be determined",
    r"no answer",
    r"\n\n",           # LLM explanation bleed-through
]

# ── style ──────────────────────────────────────────────────────────────────────
PALETTE = {"yesno": "#4C72B0", "factoid": "#DD8452",
           "list": "#55A868", "summary": "#C44E52"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for soft matching."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ── ROUGE ──────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def rouge_n(hyp: str, refs: list[str], n: int) -> dict:
    """ROUGE-N between hypothesis and list of references (takes max)."""
    if not hyp or not refs:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    hyp_tok = hyp.lower().split()
    hyp_ng  = _ngrams(hyp_tok, n)
    best_f  = 0.0
    best_pr = (0.0, 0.0)
    for ref in refs:
        ref_tok = ref.lower().split()
        ref_ng  = _ngrams(ref_tok, n)
        overlap = sum((hyp_ng & ref_ng).values())
        p = overlap / max(1, sum(hyp_ng.values()))
        r = overlap / max(1, sum(ref_ng.values()))
        f = (2 * p * r) / max(1e-9, p + r)
        if f > best_f:
            best_f = f
            best_pr = (p, r)
    return {"p": best_pr[0], "r": best_pr[1], "f": best_f}


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(hyp: str, refs: list[str]) -> dict:
    if not hyp or not refs:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    hyp_tok = hyp.lower().split()
    best_f  = 0.0
    best_pr = (0.0, 0.0)
    for ref in refs:
        ref_tok = ref.lower().split()
        lcs = _lcs(hyp_tok, ref_tok)
        p = lcs / max(1, len(hyp_tok))
        r = lcs / max(1, len(ref_tok))
        f = (2 * p * r) / max(1e-9, p + r)
        if f > best_f:
            best_f = f
            best_pr = (p, r)
    return {"p": best_pr[0], "r": best_pr[1], "f": best_f}


# ── document metrics ───────────────────────────────────────────────────────────

def doc_metrics(submitted: list[str], feedback_docs: list[dict]) -> dict:
    """
    submitted      : list of PMIDs (strings) from submission
    feedback_docs  : list of {id, golden} dicts from expert feedback
    """
    golden_set = {str(d["id"]) for d in feedback_docs if d.get("golden")}
    sub_set    = set(str(d) for d in submitted)

    if not golden_set:
        return {"precision": None, "recall": None, "f1": None, "mrr": None,
                "num_golden": 0, "num_submitted": len(sub_set),
                "true_positives": 0}

    tp = len(sub_set & golden_set)
    p  = tp / max(1, len(sub_set))
    r  = tp / max(1, len(golden_set))
    f  = (2 * p * r) / max(1e-9, p + r)

    # MRR: rank of first relevant doc in submission order
    mrr = 0.0
    for i, doc in enumerate(submitted):
        if str(doc) in golden_set:
            mrr = 1.0 / (i + 1)
            break

    return {
        "precision": p, "recall": r, "f1": f, "mrr": mrr,
        "num_golden": len(golden_set),
        "num_submitted": len(sub_set),
        "true_positives": tp,
    }


# ── snippet metrics ────────────────────────────────────────────────────────────

def snippet_metrics(submitted: list[dict], feedback_snips: list[dict]) -> dict:
    golden_texts = {normalise_text(s["text"]) for s in feedback_snips if s.get("golden")}
    sub_texts    = {normalise_text(s["text"]) for s in submitted}

    if not golden_texts:
        return {"precision": None, "recall": None, "f1": None,
                "num_golden": 0, "num_submitted": len(sub_texts), "true_positives": 0}

    tp = len(sub_texts & golden_texts)
    p  = tp / max(1, len(sub_texts))
    r  = tp / max(1, len(golden_texts))
    f  = (2 * p * r) / max(1e-9, p + r)

    return {
        "precision": p, "recall": r, "f1": f,
        "num_golden": len(golden_texts),
        "num_submitted": len(sub_texts),
        "true_positives": tp,
    }


# ── answer metrics ─────────────────────────────────────────────────────────────

def yesno_metrics(sub_exact, fb_exact) -> dict:
    """Returns 1.0 / 0.0 / None (if feedback has no answer)."""
    if not fb_exact or fb_exact == "":
        return {"correct": None}
    correct = str(sub_exact).strip().lower() == str(fb_exact).strip().lower()
    return {"correct": float(correct), "submitted": sub_exact, "golden": fb_exact}


def _normalise_entity(e: str) -> str:
    return re.sub(r"\s+", " ", e.lower().strip())


def factoid_metrics(sub_exact, fb_exact) -> dict:
    """
    sub_exact : list of [entity] single-element lists (up to 5)
    fb_exact  : list of [entity, synonym, ...] lists
    """
    if not fb_exact:
        return {"precision_at_1": None, "recall": None, "f1": None}

    # Flatten golden: set of all valid surface forms
    golden_flat = set()
    for entry in fb_exact:
        if isinstance(entry, list):
            for e in entry:
                if e and e.strip():
                    golden_flat.add(_normalise_entity(e))
        elif isinstance(entry, str) and entry.strip():
            golden_flat.add(_normalise_entity(entry))

    if not golden_flat:
        return {"precision_at_1": None, "recall": None, "f1": None}

    # Submitted entities: take first element of each inner list
    submitted_entities = []
    if isinstance(sub_exact, list):
        for entry in sub_exact:
            if isinstance(entry, list) and entry and entry[0].strip():
                submitted_entities.append(_normalise_entity(entry[0]))
            elif isinstance(entry, str) and entry.strip():
                submitted_entities.append(_normalise_entity(entry))

    # P@1: first submitted entity in golden set
    p_at_1 = float(bool(submitted_entities and submitted_entities[0] in golden_flat))

    # Broader F1 over submitted entity set
    sub_set = set(submitted_entities)
    tp = len(sub_set & golden_flat)
    p  = tp / max(1, len(sub_set))
    r  = tp / max(1, len(golden_flat))
    f  = (2 * p * r) / max(1e-9, p + r)

    return {
        "precision_at_1": p_at_1,
        "precision": p, "recall": r, "f1": f,
        "num_golden_entities": len(golden_flat),
        "num_submitted_entities": len(sub_set),
    }


def list_metrics(sub_exact, fb_exact) -> dict:
    """
    Both are lists of lists.  Each inner list contains synonyms; the
    canonical name is the first element of each golden inner list.
    """
    if not fb_exact:
        return {"precision": None, "recall": None, "f1": None}

    golden_flat = set()
    for entry in fb_exact:
        if isinstance(entry, list):
            for e in entry:
                if e and e.strip():
                    golden_flat.add(_normalise_entity(e))
        elif isinstance(entry, str) and entry.strip():
            golden_flat.add(_normalise_entity(entry))

    if not golden_flat:
        return {"precision": None, "recall": None, "f1": None}

    sub_flat = set()
    if isinstance(sub_exact, list):
        for entry in sub_exact:
            if isinstance(entry, list):
                for e in entry:
                    if e and e.strip():
                        sub_flat.add(_normalise_entity(e))
            elif isinstance(entry, str) and entry.strip():
                sub_flat.add(_normalise_entity(entry))

    tp = len(sub_flat & golden_flat)
    p  = tp / max(1, len(sub_flat))
    r  = tp / max(1, len(golden_flat))
    f  = (2 * p * r) / max(1e-9, p + r)

    return {
        "precision": p, "recall": r, "f1": f,
        "num_golden_entities": len(golden_flat),
        "num_submitted_entities": len(sub_flat),
        "true_positives": tp,
    }


def ideal_metrics(sub_ideal: str, fb_ideal) -> dict:
    """ROUGE-1/2/L between submitted string and one or more golden strings."""
    if not sub_ideal:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
                "rouge1_p": 0.0, "rouge2_p": 0.0, "rougeL_p": 0.0,
                "rouge1_r": 0.0, "rouge2_r": 0.0, "rougeL_r": 0.0}
    if isinstance(fb_ideal, str):
        refs = [fb_ideal] if fb_ideal else []
    elif isinstance(fb_ideal, list):
        refs = [r for r in fb_ideal if r]
    else:
        refs = []
    if not refs:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
                "rouge1_p": 0.0, "rouge2_p": 0.0, "rougeL_p": 0.0,
                "rouge1_r": 0.0, "rouge2_r": 0.0, "rougeL_r": 0.0}
    r1 = rouge_n(sub_ideal, refs, 1)
    r2 = rouge_n(sub_ideal, refs, 2)
    rl = rouge_l(sub_ideal, refs)
    return {
        "rouge1_f": r1["f"], "rouge1_p": r1["p"], "rouge1_r": r1["r"],
        "rouge2_f": r2["f"], "rouge2_p": r2["p"], "rouge2_r": r2["r"],
        "rougeL_f": rl["f"], "rougeL_p": rl["p"], "rougeL_r": rl["r"],
    }


# ── corpus coverage helpers ────────────────────────────────────────────────────

def corpus_coverage_metrics(submitted_docs: list[str], feedback_docs: list[dict]) -> dict:
    """
    Check how many of the golden documents from expert feedback were even
    present in the submission's candidate pool (i.e., the 549-doc corpus).
    Unlike doc_metrics(), this measures whether retrieval was possible at all,
    not just whether it succeeded.
    """
    golden_ids = {str(d["id"]) for d in feedback_docs if d.get("golden")}
    corpus_ids = {str(d) for d in submitted_docs}          # what was in our index
    reachable  = golden_ids & corpus_ids                   # golden docs we could have found

    n_golden   = len(golden_ids)
    n_corpus   = len(corpus_ids)
    n_reachable = len(reachable)

    coverage = n_reachable / max(1, n_golden)              # fraction of golden docs in corpus
    return {
        "n_golden": n_golden,
        "n_corpus": n_corpus,
        "n_reachable": n_reachable,
        "coverage": coverage,                              # 0 = none of the relevant docs indexed
    }


def is_malformed_answer(exact_answer, qtype: str) -> tuple[bool, str]:
    """
    Detect known LLM failure modes in the exact_answer field.
    Returns (is_malformed, reason).

    Note: summary questions correctly have exact_answer=[] in BioASQ format;
    we do NOT flag them as malformed.
    """
    ea_str = str(exact_answer).lower().strip()

    # Summary: empty list is the correct BioASQ format — never malformed
    if qtype == "summary":
        return False, ""

    # Empty / null (for non-summary types this is a failure)
    if not exact_answer or ea_str in ("none", "[]", ""):
        return True, "empty_answer"

    # Fallback phrases
    for pat in MALFORMED_PATTERNS:
        if re.search(pat, ea_str):
            return True, f"pattern:{pat.strip()}"

    # Factoid-specific: answer is too long (should be a short entity, not a sentence)
    if qtype == "factoid":
        if isinstance(exact_answer, list) and exact_answer:
            first = exact_answer[0]
            if isinstance(first, list) and first:
                text = first[0]
            elif isinstance(first, str):
                text = first
            else:
                text = ""
            word_count = len(str(text).split())
            if word_count > 15:
                return True, f"factoid_too_long:{word_count}_words"

    # List-specific: only one item submitted (usually means hallucination or fallback)
    if qtype == "list":
        if isinstance(exact_answer, list) and len(exact_answer) == 1:
            item = exact_answer[0]
            item_str = str(item[0] if isinstance(item, list) else item).lower()
            for pat in MALFORMED_PATTERNS:
                if re.search(pat, item_str):
                    return True, f"list_single_fallback"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# main analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse():
    sub_data    = load_json(SUB_PATH)
    fb_data     = load_json(FB_PATH)
    golden_data = load_json(GOLDEN_PATH)

    sub_map    = {q["id"]: q for q in sub_data["questions"]}
    fb_map     = {q["id"]: q for q in fb_data["questions"]}
    golden_map = {q["id"]: q for q in golden_data["questions"]}

    matched_ids = sorted(set(sub_map) & set(fb_map))
    missing_ids = sorted(set(fb_map) - set(sub_map))       # questions never submitted
    print(f"Submitted : {len(sub_map)} questions")
    print(f"Feedback  : {len(fb_map)} questions")
    print(f"Matched   : {len(matched_ids)}")
    print(f"Missing   : {len(missing_ids)} (questions not submitted at all)")

    # ── collect all corpus doc IDs (the 549-doc pool) ─────────────────────────
    corpus_doc_ids = set()
    for q in sub_data["questions"]:
        corpus_doc_ids.update(str(d) for d in q.get("documents", []))
    print(f"Unique docs in corpus: {len(corpus_doc_ids)}")

    per_question = []

    for qid in matched_ids:
        sq = sub_map[qid]
        fq = fb_map[qid]
        qtype = fq.get("type", "unknown")
        answer_ready = fq.get("answerReady", False)

        row = {
            "id":           qid,
            "type":         qtype,
            "answer_ready": answer_ready,
            "body":         fq.get("body", ""),
        }

        # ── document retrieval ────────────────────────────────────────────────
        dm = doc_metrics(sq.get("documents", []), fq.get("documents", []))
        row.update({f"doc_{k}": v for k, v in dm.items()})

        # ── corpus coverage: how many golden docs were even indexable? ─────────
        cc = corpus_coverage_metrics(list(corpus_doc_ids), fq.get("documents", []))
        row.update({f"corpus_{k}": v for k, v in cc.items()})

        # ── snippet retrieval ─────────────────────────────────────────────────
        sm = snippet_metrics(sq.get("snippets", []), fq.get("snippets", []))
        row.update({f"snip_{k}": v for k, v in sm.items()})

        # ── malformed answer detection ─────────────────────────────────────────
        sub_ea = sq.get("exact_answer")
        malformed, malformed_reason = is_malformed_answer(sub_ea, qtype)
        row["answer_malformed"]        = malformed
        row["answer_malformed_reason"] = malformed_reason

        # ── answer quality ────────────────────────────────────────────────────
        if answer_ready:
            fb_ea  = fq.get("exact_answer")
            sub_ia = sq.get("ideal_answer", "")
            fb_ia  = fq.get("ideal_answer")

            if qtype == "yesno":
                am = yesno_metrics(sub_ea, fb_ea)
                row.update({f"yesno_{k}": v for k, v in am.items()})

            elif qtype == "factoid":
                am = factoid_metrics(sub_ea, fb_ea)
                row.update({f"factoid_{k}": v for k, v in am.items()})

            elif qtype == "list":
                am = list_metrics(sub_ea, fb_ea)
                row.update({f"list_{k}": v for k, v in am.items()})

            # ideal answer for all types
            im = ideal_metrics(sub_ia, fb_ia)
            row.update({f"ideal_{k}": v for k, v in im.items()})

        per_question.append(row)

    # ── missing question summary ───────────────────────────────────────────────
    missing_rows = []
    for qid in missing_ids:
        fq = fb_map[qid]
        gq = golden_map.get(qid, {})
        missing_rows.append({
            "id":           qid,
            "type":         fq.get("type", "unknown"),
            "answer_ready": fq.get("answerReady", False),
            "body":         fq.get("body", ""),
            "golden_docs":  len([d for d in fq.get("documents", []) if d.get("golden")]),
        })
    missing_df = pd.DataFrame(missing_rows) if missing_rows else pd.DataFrame()

    df = pd.DataFrame(per_question)

    # ══════════════════════════════════════════════════════════════════════════
    # aggregate metrics
    # ══════════════════════════════════════════════════════════════════════════

    def mean_nonnull(series):
        vals = series.dropna()
        return float(vals.mean()) if len(vals) else None

    metrics = {
        "summary": {
            "total_questions_feedback":  len(fb_map),
            "total_questions_submitted": len(sub_map),
            "matched_questions":         len(matched_ids),
            "missing_questions":         len(missing_ids),
            "coverage_pct":              100 * len(matched_ids) / max(1, len(fb_map)),
            "corpus_unique_docs":        len(corpus_doc_ids),
        },
        "corpus_coverage": {},
        "document_retrieval": {},
        "snippet_retrieval": {},
        "answer_quality": {},
        "malformed_answers": {},
    }

    # ── corpus coverage aggregates ─────────────────────────────────────────────
    metrics["corpus_coverage"]["overall_coverage"]       = mean_nonnull(df["corpus_coverage"])
    metrics["corpus_coverage"]["total_golden_docs"]      = int(df["corpus_n_golden"].sum())
    metrics["corpus_coverage"]["total_reachable_docs"]   = int(df["corpus_n_reachable"].sum())
    metrics["corpus_coverage"]["questions_zero_coverage"]= int((df["corpus_coverage"] == 0.0).sum())
    metrics["corpus_coverage"]["questions_any_coverage"] = int((df["corpus_coverage"] > 0.0).sum())
    for qtype in ("yesno", "factoid", "list", "summary"):
        sub_df = df[df["type"] == qtype]
        metrics["corpus_coverage"][f"{qtype}_coverage"] = mean_nonnull(sub_df["corpus_coverage"])

    # ── malformed answer aggregates ────────────────────────────────────────────
    metrics["malformed_answers"]["total_malformed"]       = int(df["answer_malformed"].sum())
    metrics["malformed_answers"]["total_submitted"]       = len(df)
    metrics["malformed_answers"]["malformed_pct"]         = 100 * df["answer_malformed"].mean()
    for qtype in ("yesno", "factoid", "list", "summary"):
        sub_df = df[df["type"] == qtype]
        metrics["malformed_answers"][f"{qtype}_malformed"] = int(sub_df["answer_malformed"].sum())
        metrics["malformed_answers"][f"{qtype}_total"]     = len(sub_df)
    # Reason breakdown
    reasons = df[df["answer_malformed"]]["answer_malformed_reason"].value_counts().to_dict()
    metrics["malformed_answers"]["reason_counts"] = {str(k): int(v) for k, v in reasons.items()}

    # ── missing questions summary ──────────────────────────────────────────────
    if not missing_df.empty:
        metrics["missing_questions"] = {
            "total": len(missing_df),
            "by_type": missing_df["type"].value_counts().to_dict(),
            "answer_ready_missing": int(missing_df["answer_ready"].sum()),
            "total_golden_docs_missed": int(missing_df["golden_docs"].sum()),
        }
    else:
        metrics["missing_questions"] = {"total": 0}

    # overall document retrieval
    for metric in ("precision", "recall", "f1", "mrr"):
        metrics["document_retrieval"][f"overall_{metric}"] = mean_nonnull(df[f"doc_{metric}"])
    for qtype in ("yesno", "factoid", "list", "summary"):
        sub_df = df[df["type"] == qtype]
        for metric in ("precision", "recall", "f1", "mrr"):
            col = f"doc_{metric}"
            if col in sub_df.columns:
                metrics["document_retrieval"][f"{qtype}_{metric}"] = mean_nonnull(sub_df[col])

    # overall snippet retrieval
    for metric in ("precision", "recall", "f1"):
        metrics["snippet_retrieval"][f"overall_{metric}"] = mean_nonnull(df[f"snip_{metric}"])
    for qtype in ("yesno", "factoid", "list", "summary"):
        sub_df = df[df["type"] == qtype]
        for metric in ("precision", "recall", "f1"):
            col = f"snip_{metric}"
            if col in sub_df.columns:
                metrics["snippet_retrieval"][f"{qtype}_{metric}"] = mean_nonnull(sub_df[col])

    # answer quality
    ar_df = df[df["answer_ready"] == True]

    # yesno accuracy
    yn_df = ar_df[ar_df["type"] == "yesno"]
    if "yesno_correct" in yn_df.columns:
        metrics["answer_quality"]["yesno_accuracy"] = mean_nonnull(yn_df["yesno_correct"])
        metrics["answer_quality"]["yesno_n"] = int(yn_df["yesno_correct"].notna().sum())

    # factoid
    fa_df = ar_df[ar_df["type"] == "factoid"]
    for m in ("precision_at_1", "precision", "recall", "f1"):
        col = f"factoid_{m}"
        if col in fa_df.columns:
            metrics["answer_quality"][f"factoid_{m}"] = mean_nonnull(fa_df[col])
    metrics["answer_quality"]["factoid_n"] = len(fa_df)

    # list
    li_df = ar_df[ar_df["type"] == "list"]
    for m in ("precision", "recall", "f1"):
        col = f"list_{m}"
        if col in li_df.columns:
            metrics["answer_quality"][f"list_{m}"] = mean_nonnull(li_df[col])
    metrics["answer_quality"]["list_n"] = len(li_df)

    # ideal answers (ROUGE)
    for m in ("rouge1_f", "rouge2_f", "rougeL_f",
              "rouge1_p", "rouge2_p", "rougeL_p",
              "rouge1_r", "rouge2_r", "rougeL_r"):
        col = f"ideal_{m}"
        if col in ar_df.columns:
            metrics["answer_quality"][f"overall_{m}"] = mean_nonnull(ar_df[col])
    # ROUGE by type
    for qtype in ("yesno", "factoid", "list", "summary"):
        sub = ar_df[ar_df["type"] == qtype]
        for m in ("rouge1_f", "rouge2_f", "rougeL_f"):
            col = f"ideal_{m}"
            if col in sub.columns:
                metrics["answer_quality"][f"{qtype}_{m}"] = mean_nonnull(sub[col])

    # ── error taxonomy ─────────────────────────────────────────────────────────
    taxonomy = {}

    # documents: zero-precision (submitted nothing useful)
    zero_doc_p = (df["doc_precision"] == 0.0).sum()
    zero_doc_r = (df["doc_recall"] == 0.0).sum()
    taxonomy["doc_zero_precision_questions"]      = int(zero_doc_p)
    taxonomy["doc_zero_recall_questions"]         = int(zero_doc_r)
    taxonomy["doc_high_precision_low_recall"]     = int(((df["doc_precision"] >= 0.5) & (df["doc_recall"] < 0.3)).sum())
    taxonomy["doc_low_precision_high_recall"]     = int(((df["doc_precision"] < 0.3) & (df["doc_recall"] >= 0.5)).sum())
    taxonomy["doc_good_performance"]              = int(((df["doc_f1"] >= 0.5)).sum())

    # snippets
    taxonomy["snip_zero_precision_questions"]     = int((df["snip_precision"] == 0.0).sum())
    taxonomy["snip_zero_recall_questions"]        = int((df["snip_recall"] == 0.0).sum())
    taxonomy["snip_good_performance"]             = int(((df["snip_f1"] >= 0.3)).sum())

    # yesno errors
    if "yesno_correct" in df.columns:
        wrong_yn = yn_df[yn_df["yesno_correct"] == 0.0]
        taxonomy["yesno_wrong"] = int(len(wrong_yn))
        taxonomy["yesno_wrong_submitted"]  = list(wrong_yn["yesno_submitted"].tolist()) if "yesno_submitted" in wrong_yn.columns else []
        taxonomy["yesno_wrong_golden"]     = list(wrong_yn["yesno_golden"].tolist())    if "yesno_golden"    in wrong_yn.columns else []

    metrics["error_taxonomy"] = taxonomy

    # ══════════════════════════════════════════════════════════════════════════
    # save metrics JSON
    # ══════════════════════════════════════════════════════════════════════════

    out_metrics = OUT_DIR / "metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"Metrics saved → {out_metrics}")

    # ══════════════════════════════════════════════════════════════════════════
    # plots
    # ══════════════════════════════════════════════════════════════════════════

    _plot_overview_bar(metrics)
    _plot_doc_metrics_by_type(df)
    _plot_snippet_metrics_by_type(df)
    _plot_doc_score_distributions(df)
    _plot_snip_score_distributions(df)
    _plot_rouge_by_type(df)
    _plot_doc_precision_recall_scatter(df)
    _plot_yesno_confusion(df)
    _plot_factoid_list_f1_dist(df)
    _plot_doc_count_analysis(df)
    _plot_error_heatmap(df)
    _plot_f1_by_question_length(df)
    _plot_corpus_coverage(df, metrics)
    _plot_missing_and_malformed(missing_df, df, metrics)

    print(f"\nAll plots saved → {PLOT_DIR}/")
    return metrics, df


# ══════════════════════════════════════════════════════════════════════════════
# individual plot functions
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(name: str):
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  plot: {path.name}")


def _plot_overview_bar(metrics: dict):
    """Summary bar chart of the major metrics across all question types."""
    cats = {
        "Doc\nPrecision":  metrics["document_retrieval"].get("overall_precision"),
        "Doc\nRecall":     metrics["document_retrieval"].get("overall_recall"),
        "Doc\nF1":         metrics["document_retrieval"].get("overall_f1"),
        "Doc\nMRR":        metrics["document_retrieval"].get("overall_mrr"),
        "Snip\nPrecision": metrics["snippet_retrieval"].get("overall_precision"),
        "Snip\nRecall":    metrics["snippet_retrieval"].get("overall_recall"),
        "Snip\nF1":        metrics["snippet_retrieval"].get("overall_f1"),
        "YesNo\nAccuracy": metrics["answer_quality"].get("yesno_accuracy"),
        "Factoid\nP@1":    metrics["answer_quality"].get("factoid_precision_at_1"),
        "List\nF1":        metrics["answer_quality"].get("list_f1"),
        "ROUGE-1\n(all)":  metrics["answer_quality"].get("overall_rouge1_f"),
        "ROUGE-2\n(all)":  metrics["answer_quality"].get("overall_rouge2_f"),
        "ROUGE-L\n(all)":  metrics["answer_quality"].get("overall_rougeL_f"),
    }
    labels = list(cats.keys())
    values = [v if v is not None else 0.0 for v in cats.values()]

    colors = (["#4C72B0"] * 4 + ["#DD8452"] * 3 +
              ["#55A868"] * 3 + ["#C44E52"] * 3)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SynapFlow — Overall Performance Summary (vs Expert Feedback)", fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Document retrieval"),
        mpatches.Patch(color="#DD8452", label="Snippet retrieval"),
        mpatches.Patch(color="#55A868", label="Exact answers"),
        mpatches.Patch(color="#C44E52", label="Ideal answers (ROUGE)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    _savefig("01_overview_summary")


def _plot_doc_metrics_by_type(df: pd.DataFrame):
    """Grouped bar: document precision/recall/F1/MRR by question type."""
    types = ["yesno", "factoid", "list", "summary"]
    metrics_list = ["doc_precision", "doc_recall", "doc_f1", "doc_mrr"]
    labels = ["Precision", "Recall", "F1", "MRR"]

    x = np.arange(len(types))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (col, label) in enumerate(zip(metrics_list, labels)):
        vals = [df[df["type"] == t][col].mean() for t in types]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Document Retrieval Metrics by Question Type", fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    _savefig("02_doc_metrics_by_type")


def _plot_snippet_metrics_by_type(df: pd.DataFrame):
    """Grouped bar: snippet precision/recall/F1 by question type."""
    types = ["yesno", "factoid", "list", "summary"]
    metrics_list = ["snip_precision", "snip_recall", "snip_f1"]
    labels = ["Precision", "Recall", "F1"]

    x = np.arange(len(types))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (col, label) in enumerate(zip(metrics_list, labels)):
        vals = [df[df["type"] == t][col].mean() for t in types]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Snippet Retrieval Metrics by Question Type", fontweight="bold")
    ax.legend()
    _savefig("03_snippet_metrics_by_type")


def _plot_doc_score_distributions(df: pd.DataFrame):
    """Violin plots of document F1 per question type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, label in zip(axes,
                               ["doc_precision", "doc_recall", "doc_f1"],
                               ["Precision", "Recall", "F1"]):
        plot_data = []
        plot_labels = []
        for t in ["yesno", "factoid", "list", "summary"]:
            vals = df[df["type"] == t][col].dropna().tolist()
            plot_data.append(vals)
            plot_labels.append(t.capitalize())
        parts = ax.violinplot(plot_data, showmedians=True)
        for pc, t in zip(parts["bodies"], ["yesno", "factoid", "list", "summary"]):
            pc.set_facecolor(PALETTE[t])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(plot_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Doc {label} Distribution")
    fig.suptitle("Document Retrieval Score Distributions by Type", fontweight="bold")
    plt.tight_layout()
    _savefig("04_doc_score_distributions")


def _plot_snip_score_distributions(df: pd.DataFrame):
    """Violin plots of snippet F1 per question type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, label in zip(axes,
                               ["snip_precision", "snip_recall", "snip_f1"],
                               ["Precision", "Recall", "F1"]):
        plot_data = []
        plot_labels = []
        for t in ["yesno", "factoid", "list", "summary"]:
            vals = df[df["type"] == t][col].dropna().tolist()
            plot_data.append(vals)
            plot_labels.append(t.capitalize())
        parts = ax.violinplot(plot_data, showmedians=True)
        for pc, t in zip(parts["bodies"], ["yesno", "factoid", "list", "summary"]):
            pc.set_facecolor(PALETTE[t])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(plot_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Snip {label} Distribution")
    fig.suptitle("Snippet Retrieval Score Distributions by Type", fontweight="bold")
    plt.tight_layout()
    _savefig("05_snip_score_distributions")


def _plot_rouge_by_type(df: pd.DataFrame):
    """Grouped bar: ROUGE-1/2/L F1 by question type for answer-ready questions."""
    ar = df[df["answer_ready"] == True]
    types = ["yesno", "factoid", "list", "summary"]
    rouge_cols = ["ideal_rouge1_f", "ideal_rouge2_f", "ideal_rougeL_f"]
    rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    x = np.arange(len(types))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (col, label) in enumerate(zip(rouge_cols, rouge_labels)):
        vals = [ar[ar["type"] == t][col].mean() if col in ar.columns else 0.0 for t in types]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ideal Answer ROUGE Scores by Question Type", fontweight="bold")
    ax.legend()
    _savefig("06_rouge_by_type")


def _plot_doc_precision_recall_scatter(df: pd.DataFrame):
    """Precision vs Recall scatter per question, coloured by type."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for qtype in ["yesno", "factoid", "list", "summary"]:
        sub = df[df["type"] == qtype]
        ax.scatter(sub["doc_recall"], sub["doc_precision"],
                   color=PALETTE[qtype], label=qtype.capitalize(),
                   alpha=0.65, s=60, edgecolors="white", linewidths=0.5)

    # iso-F1 curves
    for f in [0.2, 0.4, 0.6, 0.8]:
        x_pts = np.linspace(0.01, 1.0, 200)
        y_pts = f * x_pts / (2 * x_pts - f + 1e-9)
        mask = (y_pts >= 0) & (y_pts <= 1)
        ax.plot(x_pts[mask], y_pts[mask], "--", color="grey", linewidth=0.7, alpha=0.5)
        ax.text(x_pts[mask][-1] + 0.01, y_pts[mask][-1], f"F1={f}", fontsize=7, color="grey")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Document Retrieval: Precision vs Recall per Question", fontweight="bold")
    ax.legend()
    _savefig("07_doc_precision_recall_scatter")


def _plot_yesno_confusion(df: pd.DataFrame):
    """Yes/No confusion matrix."""
    yn_df = df[(df["type"] == "yesno") & df["answer_ready"]]
    if "yesno_submitted" not in yn_df.columns or "yesno_golden" not in yn_df.columns:
        return

    sub_vals = yn_df["yesno_submitted"].fillna("").str.lower().str.strip()
    gold_vals = yn_df["yesno_golden"].fillna("").str.lower().str.strip()

    labels = ["yes", "no"]
    conf = np.zeros((2, 2), dtype=int)
    for s, g in zip(sub_vals, gold_vals):
        gi = 0 if g == "yes" else 1
        si = 0 if s == "yes" else 1
        conf[gi][si] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Yes", "Pred No"],
                yticklabels=["True Yes", "True No"], ax=ax)
    ax.set_title("Yes/No Answer Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    _savefig("08_yesno_confusion")


def _plot_factoid_list_f1_dist(df: pd.DataFrame):
    """Histogram of F1 scores for factoid and list answers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, qtype, col, title in [
        (axes[0], "factoid", "factoid_f1", "Factoid Answer F1 Distribution"),
        (axes[1], "list",    "list_f1",    "List Answer F1 Distribution"),
    ]:
        sub = df[(df["type"] == qtype) & df["answer_ready"]]
        if col in sub.columns:
            vals = sub[col].dropna()
            ax.hist(vals, bins=20, range=(0, 1), color=PALETTE[qtype],
                    edgecolor="white", linewidth=0.5)
            ax.axvline(vals.mean(), color="red", linestyle="--",
                       label=f"Mean={vals.mean():.3f}")
            ax.axvline(vals.median(), color="orange", linestyle=":",
                       label=f"Median={vals.median():.3f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    _savefig("09_factoid_list_f1_distributions")


def _plot_doc_count_analysis(df: pd.DataFrame):
    """Submitted docs vs golden docs: how many golden docs exist and how many we find."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Distribution of num_golden docs per question
    ax = axes[0]
    for qtype in ["yesno", "factoid", "list", "summary"]:
        sub = df[df["type"] == qtype]["doc_num_golden"].dropna()
        ax.hist(sub, bins=20, alpha=0.6, label=qtype.capitalize(),
                color=PALETTE[qtype])
    ax.set_xlabel("Number of Golden Docs")
    ax.set_ylabel("Questions")
    ax.set_title("Distribution of Golden Doc Count", fontweight="bold")
    ax.legend(fontsize=8)

    # 2. True positives found vs golden available
    ax = axes[1]
    for qtype in ["yesno", "factoid", "list", "summary"]:
        sub = df[df["type"] == qtype].dropna(subset=["doc_true_positives", "doc_num_golden"])
        ax.scatter(sub["doc_num_golden"], sub["doc_true_positives"],
                   color=PALETTE[qtype], alpha=0.6, label=qtype.capitalize(), s=40)
    max_val = df["doc_num_golden"].dropna().max()
    ax.plot([0, max_val], [0, max_val], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("Golden Docs Available")
    ax.set_ylabel("True Positives Retrieved")
    ax.set_title("Retrieved Relevant Docs (ideal = diagonal)", fontweight="bold")
    ax.legend(fontsize=8)

    # 3. Submitted count (always 10) vs true positives
    ax = axes[2]
    ax.hist(df["doc_true_positives"].dropna(), bins=range(0, 12),
            color="#4C72B0", edgecolor="white")
    ax.set_xlabel("True Positives per Question")
    ax.set_ylabel("Questions")
    ax.set_title("How Many Submitted Docs Were Golden?", fontweight="bold")

    plt.tight_layout()
    _savefig("10_doc_count_analysis")


def _plot_error_heatmap(df: pd.DataFrame):
    """
    Heat-map matrix: rows = question types, cols = metric bins.
    Colour = fraction of questions in each performance tier.
    """
    types = ["yesno", "factoid", "list", "summary"]
    metrics_cfg = [
        ("doc_f1",   "Doc F1"),
        ("snip_f1",  "Snip F1"),
        ("ideal_rouge1_f", "ROUGE-1"),
        ("ideal_rouge2_f", "ROUGE-2"),
        ("ideal_rougeL_f", "ROUGE-L"),
    ]

    # Bins: [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]
    bin_edges  = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]

    # Build a separate heatmap for each metric showing distribution across bins per type
    fig, axes = plt.subplots(1, len(metrics_cfg), figsize=(18, 5))
    for ax, (col, title) in zip(axes, metrics_cfg):
        mat = np.zeros((len(types), len(bin_labels)))
        for i, t in enumerate(types):
            sub = df[(df["type"] == t)][col].dropna()
            if len(sub) == 0:
                continue
            counts, _ = np.histogram(sub, bins=bin_edges)
            mat[i] = counts / counts.sum()

        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=bin_labels,
                    yticklabels=[t.capitalize() for t in types],
                    ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Score bin")

    fig.suptitle("Score Distribution Heat-map by Question Type and Metric", fontweight="bold")
    plt.tight_layout()
    _savefig("11_error_heatmap")


def _plot_f1_by_question_length(df: pd.DataFrame):
    """Scatter: question body length vs doc F1, snippet F1, ROUGE-1."""
    df = df.copy()
    df["q_len"] = df["body"].str.split().str.len()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("doc_f1",         "Document F1"),
        ("snip_f1",        "Snippet F1"),
        ("ideal_rouge1_f", "ROUGE-1 F1"),
    ]
    for ax, (col, label) in zip(axes, pairs):
        for qtype in ["yesno", "factoid", "list", "summary"]:
            sub = df[df["type"] == qtype].dropna(subset=["q_len", col])
            ax.scatter(sub["q_len"], sub[col], color=PALETTE[qtype],
                       alpha=0.55, s=40, label=qtype.capitalize())
        # trend line
        valid = df.dropna(subset=["q_len", col])
        if len(valid) > 5:
            z = np.polyfit(valid["q_len"], valid[col], 1)
            p = np.poly1d(z)
            xs = np.linspace(valid["q_len"].min(), valid["q_len"].max(), 100)
            ax.plot(xs, p(xs), "--", color="black", linewidth=1.2, alpha=0.6)
        ax.set_xlabel("Question Length (words)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Question Length", fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _savefig("12_f1_by_question_length")


def _plot_corpus_coverage(df: pd.DataFrame, metrics: dict):
    """
    Two panels showing the corpus coverage problem:
    1. Per-question corpus coverage (fraction of golden docs in our 549-doc index)
    2. Stacked bar: golden docs reachable vs. unreachable by question type
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: histogram of corpus_coverage per question
    ax = axes[0]
    for qtype in ["yesno", "factoid", "list", "summary"]:
        sub = df[df["type"] == qtype]["corpus_coverage"].dropna()
        ax.hist(sub, bins=20, range=(0, 1), alpha=0.65, label=qtype.capitalize(),
                color=PALETTE[qtype])
    ax.set_xlabel("Fraction of Golden Docs in Corpus")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Corpus Coverage per Question\n(fraction of relevant docs that were indexed)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    total_zero = metrics["corpus_coverage"].get("questions_zero_coverage", 0)
    total_q    = metrics["summary"]["matched_questions"]
    ax.text(0.98, 0.95, f"{total_zero}/{total_q} questions\nhave ZERO coverage",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8))

    # Panel 2: stacked bar — reachable vs unreachable golden docs per type
    ax = axes[1]
    types = ["yesno", "factoid", "list", "summary"]
    reachable = []
    unreachable = []
    for t in types:
        sub = df[df["type"] == t]
        r = sub["corpus_n_reachable"].sum()
        g = sub["corpus_n_golden"].sum()
        reachable.append(r)
        unreachable.append(g - r)
    x = np.arange(len(types))
    ax.bar(x, reachable,   label="Reachable (in corpus)", color="#55A868")
    ax.bar(x, unreachable, bottom=reachable, label="Unreachable (not indexed)", color="#C44E52", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylabel("Number of Golden Documents")
    ax.set_title("Golden Documents: Reachable vs. Not Indexed\n(by question type)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    total_r = metrics["corpus_coverage"]["total_reachable_docs"]
    total_g = metrics["corpus_coverage"]["total_golden_docs"]
    ax.text(0.98, 0.95, f"Overall: {total_r}/{total_g}\n({100*total_r/max(1,total_g):.1f}%) reachable",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.9))

    plt.tight_layout()
    _savefig("13_corpus_coverage")


def _plot_missing_and_malformed(missing_df: pd.DataFrame, df: pd.DataFrame, metrics: dict):
    """
    Two panels:
    1. Missing questions by type (never submitted)
    2. Malformed answers by type and reason
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: missing questions by type
    ax = axes[0]
    types = ["yesno", "factoid", "list", "summary"]
    missing_by_type = metrics.get("missing_questions", {}).get("by_type", {})
    submitted_by_type = df["type"].value_counts().to_dict()
    total_by_type = {t: missing_by_type.get(t, 0) + submitted_by_type.get(t, 0) for t in types}

    missing_vals   = [missing_by_type.get(t, 0) for t in types]
    submitted_vals = [submitted_by_type.get(t, 0) for t in types]
    x = np.arange(len(types))
    ax.bar(x, submitted_vals, label="Submitted", color="#4C72B0")
    ax.bar(x, missing_vals, bottom=submitted_vals, label="Not submitted", color="#C44E52", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylabel("Number of Questions")
    total_missing = metrics.get("missing_questions", {}).get("total", 0)
    total_all = metrics["summary"]["total_questions_feedback"]
    ax.set_title(f"Question Coverage: {total_all - total_missing}/{total_all} submitted "
                 f"({100*(total_all-total_missing)/total_all:.0f}%)", fontweight="bold")
    ax.legend(fontsize=9)
    for xi, (s, m) in enumerate(zip(submitted_vals, missing_vals)):
        total = s + m
        ax.text(xi, total + 0.3, f"{100*s/max(1,total):.0f}%",
                ha="center", fontsize=9, color="#1a5276")

    # Panel 2: malformed answers by reason
    ax = axes[1]
    mal = metrics.get("malformed_answers", {})
    reason_counts = mal.get("reason_counts", {})
    if reason_counts:
        labels = [r[:30] for r in reason_counts.keys()]
        counts = list(reason_counts.values())
        colors_bar = plt.cm.Set2(np.linspace(0, 0.8, len(labels)))
        bars = ax.barh(labels, counts, color=colors_bar)
        ax.set_xlabel("Number of Questions")
        ax.set_title(f"Malformed Answer Breakdown\n"
                     f"({mal.get('total_malformed',0)}/{mal.get('total_submitted',0)} answers affected)",
                     fontweight="bold")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No malformed answers detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Malformed Answer Breakdown", fontweight="bold")

    plt.tight_layout()
    _savefig("14_missing_and_malformed")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    metrics, df = analyse()

    # ── print key findings to console ──────────────────────────────────────────
    dr = metrics["document_retrieval"]
    sr = metrics["snippet_retrieval"]
    aq = metrics["answer_quality"]
    et = metrics["error_taxonomy"]

    cc  = metrics["corpus_coverage"]
    mq  = metrics.get("missing_questions", {})
    mal = metrics.get("malformed_answers", {})

    print("\n" + "═" * 60)
    print("  SUBMISSION COVERAGE")
    print("═" * 60)
    print(f"  Submitted     : {metrics['summary']['total_questions_submitted']} / "
          f"{metrics['summary']['total_questions_feedback']} questions "
          f"({metrics['summary']['coverage_pct']:.1f}%)")
    print(f"  Missing       : {mq.get('total', 0)} questions never submitted")
    if mq.get("by_type"):
        for t, n in mq["by_type"].items():
            print(f"    {t:<10}: {n} missing")

    print("\n" + "═" * 60)
    print("  CORPUS COVERAGE (root cause #1)")
    print("═" * 60)
    print(f"  Corpus size   : {metrics['summary']['corpus_unique_docs']} unique documents")
    print(f"  Golden docs   : {cc['total_golden_docs']} needed across all questions")
    print(f"  Reachable     : {cc['total_reachable_docs']} "
          f"({100*cc['total_reachable_docs']/max(1,cc['total_golden_docs']):.1f}%) were in corpus")
    print(f"  Zero coverage : {cc['questions_zero_coverage']} questions had NO golden doc in corpus")
    print(f"  Any coverage  : {cc['questions_any_coverage']} questions had ≥1 golden doc in corpus")
    for t in ("yesno", "factoid", "list", "summary"):
        cov = cc.get(f"{t}_coverage", 0) or 0
        print(f"    {t:<10}: avg corpus coverage {cov:.3f}")

    print("\n" + "═" * 60)
    print("  MALFORMED ANSWERS (root cause #3)")
    print("═" * 60)
    print(f"  Malformed     : {mal.get('total_malformed', 0)} / "
          f"{mal.get('total_submitted', 0)} submitted answers ({mal.get('malformed_pct', 0):.1f}%)")
    for t in ("yesno", "factoid", "list", "summary"):
        m = mal.get(f"{t}_malformed", 0)
        n = mal.get(f"{t}_total", 0)
        print(f"    {t:<10}: {m}/{n}")
    if mal.get("reason_counts"):
        print("  Reason breakdown:")
        for reason, cnt in sorted(mal["reason_counts"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<40}: {cnt}")

    print("\n" + "═" * 60)
    print("  DOCUMENT RETRIEVAL")
    print("═" * 60)
    print(f"  Precision : {dr['overall_precision']:.4f}")
    print(f"  Recall    : {dr['overall_recall']:.4f}")
    print(f"  F1        : {dr['overall_f1']:.4f}")
    print(f"  MRR       : {dr['overall_mrr']:.4f}")

    print("\n" + "═" * 60)
    print("  SNIPPET RETRIEVAL")
    print("═" * 60)
    print(f"  Precision : {sr['overall_precision']:.4f}")
    print(f"  Recall    : {sr['overall_recall']:.4f}")
    print(f"  F1        : {sr['overall_f1']:.4f}")

    print("\n" + "═" * 60)
    print("  ANSWER QUALITY")
    print("═" * 60)
    if aq.get("yesno_accuracy") is not None:
        print(f"  Yes/No accuracy   : {aq['yesno_accuracy']:.4f}  (n={aq.get('yesno_n',0)})")
    if aq.get("factoid_precision_at_1") is not None:
        print(f"  Factoid P@1       : {aq['factoid_precision_at_1']:.4f}  (n={aq.get('factoid_n',0)})")
    if aq.get("list_f1") is not None:
        print(f"  List F1           : {aq['list_f1']:.4f}  (n={aq.get('list_n',0)})")
    print(f"  ROUGE-1 F1 (all)  : {aq.get('overall_rouge1_f', 0):.4f}")
    print(f"  ROUGE-2 F1 (all)  : {aq.get('overall_rouge2_f', 0):.4f}")
    print(f"  ROUGE-L F1 (all)  : {aq.get('overall_rougeL_f', 0):.4f}")

    print("\n" + "═" * 60)
    print("  ERROR TAXONOMY")
    print("═" * 60)
    print(f"  Questions with zero doc precision  : {et['doc_zero_precision_questions']}")
    print(f"  Questions with zero doc recall     : {et['doc_zero_recall_questions']}")
    print(f"  High prec / low recall             : {et['doc_high_precision_low_recall']}")
    print(f"  Low prec / high recall             : {et['doc_low_precision_high_recall']}")
    print(f"  Good doc performance (F1≥0.5)      : {et['doc_good_performance']}")
    print(f"  Questions with zero snip precision : {et['snip_zero_precision_questions']}")
    print(f"  Good snip performance (F1≥0.3)     : {et['snip_good_performance']}")
    if "yesno_wrong" in et:
        print(f"  Wrong yes/no answers               : {et['yesno_wrong']}")
    print()
