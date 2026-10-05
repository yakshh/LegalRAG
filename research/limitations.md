# Limitations

These limits apply to every number in `results/summary.json`. Read the results together with this list.

## Data and questions
- **Two Acts only**: the Information Technology Act, 2000 and the Consumer Protection Act, 2019. Results may not transfer to other statutes, case law, or other jurisdictions.
- **The IT Act is the 2000 text** without the 2008 amendments. It does not contain provisions that readers may know from the current Act (for example Section 66A or 43A). Questions about them are labelled out-of-scope on purpose.
- **PDF quality**: the extracted text has small OCR-style errors (for example "tony-five days" where the original Act says a number of days). We did not correct the documents; questions avoid wording that depends on such damaged text.
- **Small, hand-written evaluation set**: 50 questions (44 answerable, 6 out-of-scope), written by one team and not reviewed by a lawyer. Large differences between methods can be seen; small differences cannot be trusted.
- **Section-level relevance labels**: a chunk counts as relevant if it belongs to a labelled section. A question may also be answerable from another section that we did not label, so Precision@K can be too low. Very long sections (for example the definitions in Section 2) are split into many chunks, and only some of them hold the answer.
- **Page numbers** are PDF page numbers (the page where a chunk starts), not the page numbers printed in the gazette.

## Models
- **Small general-purpose retrieval models** (MiniLM embedder, MS MARCO cross-encoder). They are not trained on legal text. A legal-domain or larger model could change the ranking of the methods.
- **No tuning**: chunk size, N, M, K and the RRF constant were fixed in advance and not tuned per method. Fusion weights were not tuned.
- **A hosted LLM for generation and judging** (Gemini API, free tier). Model versions can change; free-tier quotas limited how many calls could be made per day (see the model names stored with the results). The generated answers depend on the model chosen in `config.yaml`.
- **The generator and the judge are both LLMs from the same vendor.** We use two different models, but self-preference and shared biases are still possible.

## Evaluation
- **LLM-as-a-judge is not human evaluation.** The judge can be wrong, lenient or inconsistent, and it was run once per answer at temperature 0. No human scoring was done unless someone fills in `evaluation/manual_scoring_sheet.csv`.
- **Deterministic citation checks** only verify that citations match retrieved chunks and labelled sections; they do not check that the cited text actually supports the sentence.
- **Token recall** is only a rough proxy of correctness.
- **Statistics**: the permutation tests are exploratory, use 44 paired questions, and apply no correction for multiple comparisons. A difference being smaller than the standard error should not be reported as an improvement.
- **Latency** depends on the machine, on whether the model is cached, and on other programs running. Only the relative size (retrieval vs retrieval + reranking) is meaningful. Generation time depends on the API server and the network.
- **One run**: answers were generated once per question and method; no repeated runs, so run-to-run variation of the LLM is not measured.

## System
- **English only.**
- **No query rewriting, relevance check, knowledge graph or fine-tuning.** Several of the reviewed papers use these; LegalRAG does not.
- **The system gives legal information from the retrieved text only.** It is not legal advice and must not be used as such.
