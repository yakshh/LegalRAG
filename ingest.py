"""Step 1: read PDFs, split into section-aware chunks, save them to chunks.json."""
import glob
import json
import os
import re

import fitz  # PyMuPDF
import yaml

cfg = yaml.safe_load(open("config.yaml"))
SECTION = re.compile(r"^\s*(\d+[A-Z]?)\.\s+(\S.*)")  # a line like "2. Definitions.-"
MIN_CHARS = 80  # shorter pieces are table-of-contents lines, not real sections


def clean(text):
    text = text.replace("�", "-")        # broken dash characters in the PDFs
    return re.sub(r"\s+", " ", text).strip()  # one line, single spaces


def split_text(text, size, overlap):
    """Cut text into pieces of about `size` characters, always breaking at a space."""
    pieces, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            end = text.rfind(" ", start + overlap + 1, end) + 1 or end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        start = text.find(" ", start) + 1 or start   # start the next piece at a word boundary
    return pieces


def main():
    chunks = []
    for path in sorted(glob.glob(os.path.join(cfg["pdf_folder"], "*.pdf"))):
        doc_name = os.path.basename(path)
        section, buffer, buffer_page = "Preamble", "", 1

        def save_section():
            text = clean(buffer)
            if len(text) < MIN_CHARS:
                return
            for piece in split_text(text, cfg["chunk_size"], cfg["chunk_overlap"]):
                chunks.append({"id": len(chunks), "document": doc_name, "page": buffer_page,
                               "section": section, "text": piece})

        for page_no, page in enumerate(fitz.open(path), start=1):
            for line in page.get_text().splitlines():
                m = SECTION.match(line)
                if m:  # a new section starts: save the previous one first
                    save_section()
                    section, buffer, buffer_page = f"Section {m.group(1)}", "", page_no
                buffer += line + "\n"
        save_section()

    json.dump(chunks, open("chunks.json", "w"), indent=1)
    print(f"Saved {len(chunks)} chunks")


if __name__ == "__main__":
    main()
