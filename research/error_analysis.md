# Error analysis

Written from the real outputs of this project: `evaluation/retrieval_results.csv`, `evaluation/answers.json`, `evaluation/answer_results.csv`, `results/summary.json` and the tables in `results/error_cases.md` (regenerate with `python analyze_errors.py`). Question numbers (Q001 ...) are the ids in `evaluation/questions.json`.

How to read the ranks: the rank of the first relevant chunk in the top 10 (1 = best). Answers are written from the top 5 only, so a rank above 5 means the evidence never reached the LLM.

All findings are on 44 answerable and 6 out-of-scope questions with one run per method. They show *patterns*, not statistics.

## 1. Where BM25 succeeds and dense fails

Seven questions had the evidence in BM25's top 5 but not in dense's top 5 (table A in `error_cases.md`):

| Question | Section | dense | BM25 | hybrid | rerank |
|---|---|---|---|---|---|
| Q007 compensation for damaging a computer system | 43 | 6 | 2 | 3 | 1 |
| Q011 subscriber's duties for the private key | 42 | 7 | 4 | 3 | 1 |
| Q015 employee copies data from a company network | 43 | 10 | 1 | 3 | 1 |
| Q020 can the people running a company be held responsible | 85 | miss | 1 | 2 | 1 |
| Q033 punishment for not complying with a Commission order | 72 | miss | 2 | 1 | 1 |
| Q040 who may file a complaint with a District Commission | 35 | miss | 2 | miss | 2 |
| Q044 can a reseller bring a complaint as a consumer | 2 | miss | 1 | 7 | 5 |

Pattern: the questions contain the Act's own words ("private key", "Digital Signature Certificate", "company", "complaint", "District Commission", "order"). A 384-dimension embedding of a 1000-character chunk blurs such words, while BM25 rewards them directly. In this corpus BM25 alone was slightly better than dense alone on MRR (0.687 vs 0.648), Recall@5 (0.777 vs 0.705) and nDCG@5 (0.695 vs 0.634), which is consistent with statutes using fixed vocabulary. This is the case for the lexical half of hybrid retrieval.

## 2. Where dense succeeds and BM25 fails

Five questions (table B): Q001 (electronic record), Q025 (defect), Q027 (product liability), Q014 (excluded documents) and Q030 (monetary jurisdiction). Several are **definition** questions whose key term appears in many chunks of the long definitions section, so BM25 term statistics do not point to the right chunk, while the embedding of the sentence-like question matched the definition's wording ("X means ..."). Q014 and Q030 ask in everyday words ("excluded", "monetary jurisdictions"), where dense retrieval handles the paraphrase.

## 3. Hybrid fusion helps and hurts

Hybrid (RRF) beat dense on Recall@5 (0.830 vs 0.705) and Hit@5 (0.864 vs 0.750), but the paired tests do not separate them (for example Hit@5 p = 0.13, Recall@5 p = 0.07, MRR p = 0.46). Hybrid ranked the evidence **lower** than dense in six questions (table E): Q001, Q014, Q025, Q027, Q030, Q032. In these, BM25's weak list pulled the fused order down. The ablation shows interleaving without RRF (Dense + BM25) reached Recall@5 0.811 and MRR 0.688, while RRF reached Recall@5 0.830 and MRR 0.679: RRF added a little recall and no ranking gain.

## 4. Reranking removes irrelevant chunks and also demotes useful ones

**Helped** (table C, 10 questions where the evidence moved up by 2 or more ranks): Q003 (4 to 1), Q004 (miss to 3), Q007 (3 to 1), Q011 (3 to 1), Q014 (miss to 2), Q015 (3 to 1), Q026 (4 to 1). The cross-encoder reads the question and the chunk together, so it prefers the chunk that actually answers the question over chunks that merely share words.

**Hurt** (table D): Q001 ("define an electronic record") went from rank 5 in hybrid to a miss after reranking, and the answer then refused (false refusal, evidence not in the context). Four other cases are small (Q018, Q024, Q028, Q039: rank 1 to rank 2) and did not change the answer. The cross-encoder is a general MS MARCO model, not trained on statutes, so it can misjudge definition-style chunks.

Net effect: MRR 0.781 vs 0.679 (hybrid) and 0.648 (dense). Reranking cost about 0.6 s per question (612 ms vs about 14 ms), roughly 44 times slower for retrieval, and a one-off model load.

## 5. Multi-section questions (table G)

Questions that need several sections are harder because K = 5 chunks must cover all of them.
- Q030 (three monetary limits, Sections 34, 47, 58): Recall@5 0.33 for dense and hybrid, 0.67 for reranking. All three methods answered "insufficient evidence", although part of the evidence was present.
- Q032 (appeal route, Sections 41, 51, 67): Recall@5 0.67 for all methods and all three refused.
- Q035 (Sections 21 and 89): hybrid 0.50, dense and rerank 1.00; the hybrid answer refused.
- Q018 and Q041: dense missed one of two sections.

A strict prompt that refuses when a link is missing is the safe behaviour for law, but it produces false refusals when only part of the evidence was retrieved. A larger K or per-section retrieval would help (future work).

## 6. Ambiguous legal terminology: definitions

Two definition questions were **not retrieved by any main method**: Q002 (who is a subscriber) and Q023 (who is a consumer). All three methods gave a false refusal. Cause: the definitions section is very long (Section 2 is split into 9 chunks in the IT Act and 30 in the Consumer Protection Act), and the single definition is a small part of a 1000-character chunk full of other definitions. Neither the embedding nor BM25 singles it out. Definition-level chunking (one chunk per defined term) would likely fix this; we did not try it, so this is a hypothesis.

## 7. Insufficient context and false refusals

False refusals on answerable questions: dense 14 of 44 (31.8%), hybrid 9 (20.5%), hybrid + reranking 6 (13.6%). In `error_cases.md` table H, the evidence was **missing from the five chunks sent to the LLM** in 10 of the 14 dense refusals, 6 of 9 hybrid refusals and 3 of 6 reranked refusals. The rest happened although some evidence was present (mostly the multi-section questions above). So most refusals are retrieval failures, and better retrieval removed many of them.

## 8. Hallucination and citations

- **Out-of-scope questions:** all 6 were refused by all three methods (100%). No made-up answer was produced, including for the Section 66A question, which is not in the 2000 text.
- **Invented or wrong citations:** none. Every citation in every answer matched a retrieved chunk (citation validity 1.00 for all methods). Citation precision (the cited section is a labelled relevant section) was 1.00 (dense), 0.97 (hybrid) and 0.98 (reranking): the small gaps are answers that also cited a neighbouring section.
- **Answers that looked supported but were wrong:** none were found by the judge (table K is empty; the judge gave correctness 0 or 1 only to refusals).
- Caveat: these conclusions come from one LLM judge (`gemini-3.1-flash-lite`) and one run. The judge gave faithfulness 3 to all 150 answers, so faithfulness could not separate the methods; this may reflect extractive, well-grounded answers, but it also means the judge may be lenient. A manual check of a sample (`--manual-sheet`) is recommended.

## 9. Summary of failure modes

| Failure mode | Seen in | Effect |
|---|---|---|
| Question words that match the Act's wording but are blurred in embeddings | Q007, Q011, Q015, Q020, Q033, Q040 | dense misses, BM25 finds |
| Definitions inside a very long, many-chunk section | Q001, Q002, Q004, Q023, Q044 | evidence not retrieved, false refusal |
| Multi-section questions | Q030, Q032, Q035, Q018 | partial evidence, refusal |
| Cross-encoder demotes a relevant definition chunk | Q001 | false refusal after reranking |
| Hybrid fusion lowers a rank that dense had right | Q001, Q014, Q025, Q027, Q030, Q032 | smaller gain than expected |
| Hallucinated answers, invented citations | none found | — |
