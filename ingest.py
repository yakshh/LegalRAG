import bisect
import glob
import json
import logging
import os
import re
from typing import List, Tuple

import fitz  # PyMuPDF
import yaml

log = logging.getLogger("ingest")

SECTION = re.compile(r"^\s*(\d+)([A-Z]{0,2})\.\s+([A-Z][a-z].*)")  # heading line such as "43. Penalty ..."
FOOTNOTE = re.compile(r"(Ins|Subs|Omitted|Added|Rep|The words|See|Vide)\b")  # "1. Ins. by Act 33 of 2021"
SCHEDULE = re.compile(r"^\s*THE\s+(\w+)\s+SCHEDULE\b", re.I)  # "THE SECOND SCHEDULE"
TOC_PAGES = 5   # a table of contents (if any) is on the first pages only
MIN_CHARS = 80  # shorter pieces are leftovers of the table of contents, not real sections


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean(text: str) -> str:
    text = text.replace(chr(0xFFFD), "-")     # broken dash characters in the PDFs
    return re.sub(r"\s+", " ", text).strip()  # single line, single spaces


def split_text(text: str, size: int, overlap: int) -> List[Tuple[int, str]]:
    """Cut text into pieces of about `size` characters, always breaking at a space.
    Returns (start_offset, piece) pairs."""
    pieces, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            end = text.rfind(" ", start + overlap + 1, end) + 1 or end
        pieces.append((start, text[start:end].strip()))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        start = text.find(" ", start) + 1 or start   # start the next piece at a word boundary
    return pieces


def parse_pdf(path: str, chunk_size: int, overlap: int) -> Tuple[List[dict], dict]:
    """Return (chunks, stats) for one PDF. Sections are detected from numbered heading lines."""
    doc_name = os.path.basename(path)
    chunks: List[dict] = []
    sections: List[str] = []
    state = {"section": "Preamble", "lines": [], "last": 0, "suffix": ""}  # lines = [(page, text)]

    def save_section():
        text, offsets = "", []          # offsets[i] = (character offset, page)
        for page_no, line in state["lines"]:
            line = clean(line)
            if line:
                offsets.append((len(text), page_no))
                text += line + " "
        text = text.strip()
        if len(text) < MIN_CHARS:
            return
        if state["section"] not in sections:
            sections.append(state["section"])
        starts = [o for o, _ in offsets]
        for start, piece in split_text(text, chunk_size, overlap):
            page = offsets[bisect.bisect_right(starts, start) - 1][1]
            chunks.append({"document": doc_name, "page": page, "section": state["section"], "text": piece})

    def start_new(section):
        save_section()
        state.update(section=section, lines=[])

    pdf = fitz.open(path)
    for page_no, page in enumerate(pdf, start=1):
        for line in page.get_text().splitlines():
            sched = SCHEDULE.match(line)
            m = SECTION.match(line)
            if m and FOOTNOTE.match(m.group(3)):
                m = None
            if sched:                       # schedules (amendments to other laws) are not sections
                start_new(f"{sched.group(1).title()} Schedule")
                state["last"] = 10 ** 6     # no more numbered sections after a schedule starts
            elif m and not state["section"].endswith("Schedule"):
                num, suffix = int(m.group(1)), m.group(2)
                if num == 1 and not suffix and state["last"] > 1 and page_no <= TOC_PAGES:
                    # numbering restarts: everything so far was the table of contents
                    keep = next((i for i, (_, t) in enumerate(state["lines"]) if "An Act to" in t), None)
                    chunks.clear()
                    sections.clear()
                    state.update(section="Preamble", last=0, suffix="",
                                 lines=state["lines"][keep:] if keep is not None else [])
                is_suffix_section = num == state["last"] and suffix and suffix > state["suffix"]
                is_next_section = state["last"] < num <= state["last"] + 2 and not suffix
                if is_suffix_section or is_next_section:
                    start_new(f"Section {num}{suffix}")
                    state["last"], state["suffix"] = num, suffix
            state["lines"].append((page_no, line))
    save_section()

    stats = {"document": doc_name, "pages": len(pdf), "sections": len(sections), "chunks": len(chunks)}
    return chunks, stats


def ingest(cfg: dict) -> Tuple[List[dict], List[dict]]:
    """Read all PDFs of cfg['documents_dir']. Returns (chunks, per-document stats)."""
    paths = sorted(glob.glob(os.path.join(cfg["documents_dir"], "*.pdf")))
    if not paths:
        raise SystemExit(f"No PDF files found in '{cfg['documents_dir']}/'. See data/README.md.")
    all_chunks: List[dict] = []
    all_stats: List[dict] = []
    for path in paths:
        chunks, stats = parse_pdf(path, cfg["chunk_size"], cfg["chunk_overlap"])
        log.info("%-32s pages=%-3d sections=%-3d chunks=%d", stats["document"], stats["pages"],
                 stats["sections"], stats["chunks"])
        all_chunks += chunks
        all_stats.append(stats)
    for i, c in enumerate(all_chunks):
        c["id"] = i
    return all_chunks, all_stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    chunks, stats = ingest(cfg)
    with open(cfg["chunks_file"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=1)
    log.info("Loaded documents: %d", len(stats))
    log.info("Pages processed: %d", sum(s["pages"] for s in stats))
    log.info("Sections detected: %d", sum(s["sections"] for s in stats))
    log.info("Chunks created: %d (chunk_size=%d, overlap=%d)", len(chunks), cfg["chunk_size"],
             cfg["chunk_overlap"])
    log.info("Saved to %s", cfg["chunks_file"])


if __name__ == "__main__":
    main()
