from ingest import clean, ingest, parse_pdf, split_text


def test_clean_collapses_whitespace_and_fixes_dashes():
    assert clean("a \n  b\t c" + chr(0xFFFD) + "d") == "a b c-d"


def test_split_text_breaks_at_spaces_and_overlaps():
    text = " ".join(f"word{i}" for i in range(200))
    pieces = split_text(text, size=100, overlap=20)
    assert len(pieces) > 5
    for start, piece in pieces:
        assert len(piece) <= 100
        assert not piece.startswith(" ")
        assert text[start:start + len(piece)] == piece          # offsets are right
        assert piece.split()[-1].startswith("word")              # last word is not cut in half
    # consecutive pieces overlap
    assert pieces[1][0] < pieces[0][0] + len(pieces[0][1])


def test_split_text_short_text_is_one_piece():
    assert split_text("short text", 1000, 100) == [(0, "short text")]


def test_parse_pdf_detects_sections_pages_and_schedule(sample_pdf):
    chunks, stats = parse_pdf(sample_pdf, chunk_size=1000, overlap=100)
    sections = list(dict.fromkeys(c["section"] for c in chunks))
    assert sections == ["Preamble", "Section 1", "Section 2", "Section 3", "First Schedule"]
    assert stats["pages"] == 3 and stats["sections"] == 5 and stats["chunks"] == len(chunks)
    by_section = {c["section"]: c for c in chunks}
    assert by_section["Section 1"]["page"] == 2
    assert by_section["Section 3"]["page"] == 3
    assert all(c["document"] == "TestAct.pdf" for c in chunks)


def test_table_of_contents_is_dropped(sample_pdf):
    chunks, _ = parse_pdf(sample_pdf, 1000, 100)
    assert not any("ARRANGEMENT OF SECTIONS" in c["text"] for c in chunks)
    assert all(c["page"] >= 2 for c in chunks)


def test_footnote_line_is_not_a_section_heading(sample_pdf):
    chunks, _ = parse_pdf(sample_pdf, 1000, 100)
    sec3 = " ".join(c["text"] for c in chunks if c["section"] == "Section 3")
    assert "Ins. by Act 5 of 2021" in sec3          # stays inside Section 3 instead of starting a new section


def test_chunk_size_and_overlap_are_configurable(sample_pdf):
    small, _ = parse_pdf(sample_pdf, chunk_size=120, overlap=20)
    big, _ = parse_pdf(sample_pdf, chunk_size=1000, overlap=20)
    assert len(small) > len(big)
    assert all(len(c["text"]) <= 120 for c in small)


def test_ingest_reads_all_pdfs_in_folder(tmp_path, sample_pdf):
    import shutil
    folder = tmp_path / "docs"
    folder.mkdir()
    shutil.copy(sample_pdf, folder / "A.pdf")
    shutil.copy(sample_pdf, folder / "B.pdf")
    chunks, stats = ingest({"documents_dir": str(folder), "chunk_size": 1000, "chunk_overlap": 100})
    assert {c["document"] for c in chunks} == {"A.pdf", "B.pdf"}
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    assert len(stats) == 2


def test_ingest_is_reproducible(sample_pdf):
    first, _ = parse_pdf(sample_pdf, 1000, 100)
    second, _ = parse_pdf(sample_pdf, 1000, 100)
    assert first == second
