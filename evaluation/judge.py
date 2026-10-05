"""LLM-as-a-judge for generated legal answers.

This is LLM-BASED evaluation, not human evaluation. The same judge model, the same fixed prompt and the
same inputs are used for Dense, Hybrid and Hybrid + Reranking. The judge is never told which retrieval
method produced an answer, and it cannot change the experiment: it only scores saved answers.

Scores are integers 0-3:
    0 = poor   1 = partially correct   2 = mostly correct   3 = fully correct
"""
import json
import re
from typing import Dict, List, Optional

from generate import GeminiProvider, LLMProvider, cfg, format_context

SCORE_KEYS = ["correctness", "faithfulness", "relevance", "citation_correctness"]

JUDGE_PROMPT = """You are a strict evaluator of answers produced by a legal question-answering system.
You receive a question, the expected (reference) answer, the generated answer, the retrieved context the
system was allowed to use, and the citations found in the generated answer.

Score each criterion with an integer from 0 to 3 (0 = poor, 1 = partially correct, 2 = mostly correct,
3 = fully correct):

- correctness: does the generated answer agree with the expected answer (facts, numbers, conditions)?
  0 = wrong or contradicts it, 1 = some key facts only, 2 = mostly correct with a minor omission or error,
  3 = fully correct and complete.
- faithfulness: is every claim in the generated answer supported by the retrieved context?
  0 = mostly unsupported or invented, 1 = several unsupported claims, 2 = one minor unsupported detail,
  3 = every claim is supported by the context.
- relevance: does the answer address the question that was asked?
  0 = off topic, 1 = weakly related, 2 = mostly addresses it, 3 = directly and fully addresses it.
- citation_correctness: do the citations (document, page, section) point to the context passages that
  support the statements, and are none invented?
  0 = no citations for factual claims, or citations of things not in the context, 1 = mostly wrong,
  2 = mostly right, 3 = all correct.

Special case. If the expected answer says there is insufficient evidence, the question cannot be answered
from the documents. Then a generated answer that clearly says there is insufficient evidence is fully
correct (3 for all four criteria). An answer that invents an answer to such a question gets 0 for
correctness and faithfulness. If the question IS answerable but the system answered with
"insufficient evidence", correctness is 0, faithfulness is 3 and citation_correctness is 0.

Return ONLY a JSON object with these keys:
{{"correctness": 0-3, "faithfulness": 0-3, "relevance": 0-3, "citation_correctness": 0-3, "reason": "one or two sentences"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{answer}

CITATIONS FOUND IN THE GENERATED ANSWER:
{citations}

RETRIEVED CONTEXT:
{context}
"""


def build_judge_prompt(record: dict) -> str:
    cites = "; ".join(f"{c['document']}, p. {c['page']}, {c['section']}" for c in record["citations"]) or "none"
    return JUDGE_PROMPT.format(question=record["question"], expected_answer=record["expected_answer"],
                               answer=record["answer"], citations=cites,
                               context=format_context(record["retrieved_context"]))


def parse_judgement(text: str) -> Dict[str, object]:
    """Parse the judge's JSON reply. Scores are clamped to 0-3; a missing score becomes None."""
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0)) if match else {}
    out: Dict[str, object] = {}
    for key in SCORE_KEYS:
        try:
            out[key] = max(0, min(3, int(data[key])))
        except (KeyError, TypeError, ValueError):
            out[key] = None
    out["reason"] = str(data.get("reason", ""))
    return out


def judge_answer(record: dict, llm: Optional[LLMProvider] = None) -> Dict[str, object]:
    """Score one answer record (question, expected_answer, answer, citations, retrieved_context)."""
    llm = llm or GeminiProvider(cfg["judge_model"])
    return parse_judgement(llm.generate(build_judge_prompt(record), json_mode=True))
