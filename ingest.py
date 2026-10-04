"""Step 1: read PDFs, split into section-aware chunks, save them to chunks.json."""
import glob
import json
import os
import re

import fitz  # PyMuPDF
import yaml

cfg = yaml.safe_load(open("config.yaml"))
SECTION = re.compile(r"^\s*(\d+[A-Z]?)\.\s+(\S.*)")  # a line like "2. Definitions.-"


def split_text(text, size, overlap):
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


def main():
    chunks = []
    for path in sorted(glob.glob(os.path.join(cfg["pdf_folder"], "*.pdf"))):
        doc_name = os.path.basename(path)
        section, buffer, buffer_page = "Preamble", "", 1

        def save_section():
            for piece in split_text(buffer.strip(), cfg["chunk_size"], cfg["chunk_overlap"]):
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
