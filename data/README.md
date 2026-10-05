# Legal documents (`data/`)

Put the Act PDFs in this folder. `python ingest.py` reads **every** `*.pdf` here, so more Acts can be added later.

## Documents used in the experiments

| Expected file name | Document | Where to get it |
|---|---|---|
| `ITAct2000.pdf` | The Information Technology Act, 2000 (Act No. 21 of 2000), as originally enacted | [AICTE copy](https://aicte.gov.in/sites/default/files/itact2000.pdf) or [India Code](https://www.indiacode.nic.in/handle/123456789/13683) |
| `ConsumerProtectionAct2019.pdf` | The Consumer Protection Act, 2019 (Act No. 35 of 2019) | [India Code](https://www.indiacode.nic.in/handle/123456789/16939) |

Download in a browser (the sites block scripted downloads) and save with exactly the file names above.
The evaluation questions in `evaluation/questions.json` refer to these two file names.

Important:
- The IT Act PDF is the **2000 text, without the 2008 amendments**. Provisions added later (for example Sections 43A or 66A) are not in it, and the evaluation treats questions about them as out-of-scope.
- The documents are used unchanged. Nothing in them is edited or rewritten.
- If a document is missing, the system does not pretend it was evaluated: `ingest.py` only indexes the PDFs that are present, and `python validate_dataset.py` reports every question whose source document is not in the index.
- The folder is listed in `.gitignore`, so the PDFs are not committed to git. They are included in `LegalRAG-CIPAT-FINAL.zip` only because both Acts are public government documents.

## How ingestion works

`python ingest.py` (settings in `config.yaml`):

1. Opens each PDF with PyMuPDF and reads it page by page.
2. Detects a table of contents on the first pages and drops it.
3. Detects section headings (lines such as `43. Penalty for damage to computer...`) and ignores gazette footnotes such as `1. Ins. by Act 33 of 2021`. Schedules are kept as `First Schedule`, `Second Schedule`, and so on.
4. Cleans the text (single spaces, broken dash characters fixed).
5. Splits every section into chunks of `chunk_size` characters with `chunk_overlap` overlap, cutting only at spaces.
6. Writes `chunks.json`. Every chunk stores `document`, `page` (the PDF page where the chunk starts), `section` and `text`.

It prints a log: documents loaded, pages processed, sections detected and chunks created.
