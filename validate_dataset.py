"""Checks evaluation/questions.json against the ingested documents.

    python validate_dataset.py

Run `python ingest.py` first (the section and page checks use chunks.json). Exit code 0 = valid, 1 = errors.
"""
import collections
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml

from evaluation.metrics import normalize

REQUIRED = ["question_id", "question", "source_document", "source_section", "source_page", "expected_answer",
            "relevant_sections", "difficulty", "category", "answerable"]
DIFFICULTIES = {"easy", "medium", "hard"}
CATEGORIES = {"definition", "section_specific", "legal_obligation", "penalty_provision", "applicability",
              "scenario_based", "comparison", "multi_section", "exact_terminology", "semantic_understanding",
              "out_of_scope"}


def validate(questions: list, chunks: List[dict], page_counts: Optional[Dict[str, int]] = None) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). `chunks` gives the valid documents, sections and the pages of each section;
    `page_counts` (document -> number of PDF pages) is optional."""
    errors: List[str] = []
    warnings: List[str] = []
    sections = collections.defaultdict(set)          # document -> sections that exist
    section_pages = collections.defaultdict(set)     # (document, section) -> pages of its chunks
    for c in chunks:
        sections[c["document"]].add(c["section"])
        section_pages[(c["document"], c["section"])].add(c["page"])
    if not isinstance(questions, list) or not questions:
        return ["questions file must contain a non-empty JSON list"], warnings

    ids = collections.Counter(q.get("question_id") for q in questions if isinstance(q, dict))
    for qid, n in ids.items():
        if n > 1:
            errors.append(f"duplicate question_id {qid} ({n} times)")

    texts = collections.Counter(q.get("question", "").strip().lower() for q in questions if isinstance(q, dict))
    for text, n in texts.items():
        if n > 1:
            warnings.append(f"duplicate question text: {text[:60]!r}")

    for q in questions:
        if not isinstance(q, dict):
            errors.append("an entry is not a JSON object")
            continue
        qid = q.get("question_id", "<no id>")
        missing = [f for f in REQUIRED if f not in q]
        if missing:
            errors.append(f"{qid}: missing fields {missing}")
            continue
        if not str(q["question"]).strip():
            errors.append(f"{qid}: empty question")
        if not str(q["expected_answer"]).strip():
            errors.append(f"{qid}: missing expected_answer")
        if q["difficulty"] not in DIFFICULTIES:
            errors.append(f"{qid}: difficulty must be one of {sorted(DIFFICULTIES)}, got {q['difficulty']!r}")
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: unknown category {q['category']!r}")
        if not isinstance(q["relevant_sections"], list):
            errors.append(f"{qid}: relevant_sections must be a list")
            continue

        if not q["answerable"]:                       # out-of-scope question
            if q["source_document"] or q["source_section"] or q["relevant_sections"]:
                errors.append(f"{qid}: an unanswerable question must have no source document/section")
            continue

        doc, sec, page = q["source_document"], q["source_section"], q["source_page"]
        if doc not in sections:
            errors.append(f"{qid}: invalid document {doc!r} (not in chunks.json: {sorted(sections)})")
            continue
        if sec not in sections[doc]:
            errors.append(f"{qid}: source section {sec!r} does not exist in {doc}")
        if not q["relevant_sections"]:
            errors.append(f"{qid}: relevant_sections is empty")
        for rs in q["relevant_sections"]:
            if rs not in sections[doc]:
                errors.append(f"{qid}: relevant section {rs!r} does not exist in {doc}")
        if sec not in q["relevant_sections"]:
            warnings.append(f"{qid}: source_section is not listed in relevant_sections")
        phrases = q.get("evidence_phrases", {})
        if not isinstance(phrases, dict):
            errors.append(f"{qid}: evidence_phrases must be an object {{section: phrase}}")
            phrases = {}
        for ps, phrase in phrases.items():
            if ps not in q["relevant_sections"]:
                errors.append(f"{qid}: evidence phrase for {ps!r}, which is not in relevant_sections")
            elif not any(normalize(phrase) in normalize(c["text"]) for c in chunks
                         if c["document"] == doc and c["section"] == ps):
                errors.append(f"{qid}: evidence phrase not found in any chunk of {ps} ({phrase[:50]!r})")
        if page is not None:
            if not isinstance(page, int) or page < 1:
                errors.append(f"{qid}: source_page must be a positive integer or null, got {page!r}")
            elif page_counts and doc in page_counts and page > page_counts[doc]:
                errors.append(f"{qid}: source_page {page} is beyond the {page_counts[doc]} pages of {doc}")
            elif sec in sections[doc] and page not in section_pages[(doc, sec)]:
                errors.append(f"{qid}: {sec} has no text on page {page} of {doc} "
                              f"(pages: {sorted(section_pages[(doc, sec)])})")
    return errors, warnings


def main() -> int:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    for path in (cfg["questions_file"], cfg["chunks_file"]):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found" + (" (run python ingest.py first)" if "chunks" in path else ""))
            return 1
    try:
        questions = json.load(open(cfg["questions_file"], encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: {cfg['questions_file']} is not valid JSON: {e}")
        return 1
    chunks = json.load(open(cfg["chunks_file"], encoding="utf-8"))

    page_counts = {}
    try:
        import fitz
        for p in glob.glob(os.path.join(cfg["documents_dir"], "*.pdf")):
            page_counts[os.path.basename(p)] = len(fitz.open(p))
    except Exception:
        pass

    errors, warnings = validate(questions, chunks, page_counts)

    print(f"Dataset: {cfg['questions_file']}")
    print(f"Questions: {len(questions)}  (answerable: {sum(bool(q.get('answerable')) for q in questions)}, "
          f"out-of-scope: {sum(not q.get('answerable') for q in questions)})")
    for field in ("category", "difficulty", "source_document"):
        counts = collections.Counter(str(q.get(field)) for q in questions)
        print(f"  by {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Documents in chunks.json: {sorted({c['document'] for c in chunks})}")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("RESULT:", "INVALID" if errors else "VALID", f"({len(errors)} errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
