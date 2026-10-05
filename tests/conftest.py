import os
import sys

import fitz
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the project reads config.yaml and chunks.json from its own folder


@pytest.fixture(scope="session")
def cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def make_pdf(path, pages):
    """Write a small PDF: `pages` is a list of pages, each a list of text lines."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((50, y), line, fontsize=9)
            y += 14
    doc.save(path)
    doc.close()


@pytest.fixture()
def sample_pdf(tmp_path):
    """A tiny Act with a table of contents, 3 sections, a footnote line and a schedule."""
    pages = [
        ["ARRANGEMENT OF SECTIONS", "1. Short title", "2. Definitions", "3. Penalty"],
        ["THE TEST ACT, 2020",
         "An Act to test the ingestion code of this project in a careful way and",
         "to prove that the preamble text before section 1 is kept as well.",
         "1. Short title. This Act may be called the Test Act, 2020 and it applies",
         "to the whole of the country for the purpose of testing the code.",
         "2. Definitions. In this Act the word thing means any thing that can be",
         "tested by a student of computer engineering in a unit test."],
        ["3. Penalty. Whoever breaks the test shall be punished with a fine which may",
         "extend to one hundred rupees and with nothing else under this Act.",
         "1. Ins. by Act 5 of 2021, s. 3 (w.e.f. 1-1-2021).",
         "THE FIRST SCHEDULE",
         "1. In section 3, for the word fine the word penalty shall be substituted in the old law."],
    ]
    path = str(tmp_path / "TestAct.pdf")
    make_pdf(path, pages)
    return path
