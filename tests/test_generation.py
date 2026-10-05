from evaluation.judge import SCORE_KEYS, build_judge_prompt, judge_answer, parse_judgement
from evaluation.metrics import citation_metrics
from generate import (FALLBACK, INSUFFICIENT, SUPPORTED, LLMProvider, answer_question, build_prompt,
                      extract_citations, is_insufficient, retry_seconds)

CHUNKS = [
    {"document": "ITAct2000.pdf", "page": 15, "section": "Section 43", "text": "Compensation not exceeding one crore."},
    {"document": "ITAct2000.pdf", "page": 20, "section": "Section 65", "text": "Imprisonment up to three years."},
]


class FakeLLM(LLMProvider):
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        return self.reply


# ---------------- citations ----------------
def test_extract_citations_single_and_multiple():
    text = ("Compensation up to one crore [ITAct2000.pdf, p. 15, Section 43]. Also "
            "[ConsumerProtectionAct2019.pdf, p. 28, Section 69] and [ITAct2000.pdf, p. 3, Section 2].")
    cites = extract_citations(text)
    assert cites == [{"document": "ITAct2000.pdf", "page": 15, "section": "Section 43"},
                     {"document": "ConsumerProtectionAct2019.pdf", "page": 28, "section": "Section 69"},
                     {"document": "ITAct2000.pdf", "page": 3, "section": "Section 2"}]


def test_extract_citations_none_and_schedule():
    assert extract_citations("No citation here.") == []
    assert extract_citations("[ITAct2000.pdf, p. 27, First Schedule]")[0]["section"] == "First Schedule"


def test_citation_metrics_valid_and_correct():
    q = {"source_document": "ITAct2000.pdf", "relevant_sections": ["Section 43"]}
    cites = [{"document": "ITAct2000.pdf", "page": 15, "section": "Section 43"}]
    m = citation_metrics(cites, CHUNKS, q)
    assert m == {"cite_present": 1.0, "cite_valid": 1.0, "cite_precision": 1.0, "cite_recall": 1.0}


def test_citation_to_a_section_that_was_not_retrieved_is_invalid():
    q = {"source_document": "ITAct2000.pdf", "relevant_sections": ["Section 43"]}
    cites = [{"document": "ITAct2000.pdf", "page": 15, "section": "Section 43"},
             {"document": "ITAct2000.pdf", "page": 99, "section": "Section 99"}]
    m = citation_metrics(cites, CHUNKS, q)
    assert m["cite_valid"] == 0.5 and m["cite_precision"] == 0.5 and m["cite_recall"] == 1.0


def test_citation_completeness_for_multi_section_question():
    q = {"source_document": "ITAct2000.pdf", "relevant_sections": ["Section 43", "Section 65"]}
    cites = [{"document": "ITAct2000.pdf", "page": 15, "section": "Section 43"}]
    assert citation_metrics(cites, CHUNKS, q)["cite_recall"] == 0.5


def test_no_citations_gives_zero_scores():
    q = {"source_document": "ITAct2000.pdf", "relevant_sections": ["Section 43"]}
    assert citation_metrics([], CHUNKS, q)["cite_present"] == 0.0


# ---------------- insufficient evidence / grounding ----------------
def test_no_retrieved_chunks_returns_fallback_without_calling_llm():
    llm = FakeLLM("should not be used")
    result = answer_question("anything?", [], llm)
    assert result["status"] == INSUFFICIENT and result["answer"] == FALLBACK
    assert llm.prompts == []


def test_llm_fallback_reply_is_marked_insufficient():
    result = answer_question("Who is the PM?", CHUNKS, FakeLLM(FALLBACK))
    assert result["status"] == INSUFFICIENT and result["citations"] == []


def test_supported_answer_has_citations():
    reply = "Compensation up to one crore rupees [ITAct2000.pdf, p. 15, Section 43]."
    result = answer_question("Penalty?", CHUNKS, FakeLLM(reply))
    assert result["status"] == SUPPORTED
    assert result["citations"][0]["section"] == "Section 43"


def test_is_insufficient_is_case_insensitive():
    assert is_insufficient("INSUFFICIENT EVIDENCE in the documents")
    assert not is_insufficient("The penalty is one crore.")


def test_prompt_contains_rules_context_and_question():
    prompt = build_prompt("What is the penalty?", CHUNKS)
    assert "ONLY the retrieved context" in prompt
    assert "Do not invent" in prompt
    assert FALLBACK in prompt
    assert "[ITAct2000.pdf, p. 15, Section 43]" in prompt and "Compensation not exceeding one crore." in prompt
    assert prompt.rstrip().endswith("Answer:") and "What is the penalty?" in prompt


def test_same_prompt_template_for_every_method():
    # the prompt depends only on question + chunks, never on the retrieval method
    assert build_prompt("q", CHUNKS) == build_prompt("q", list(CHUNKS))


def test_retry_seconds_parsing():
    assert retry_seconds("Please retry in 16.4s") == 16.4
    assert retry_seconds("Please retry in 1h2m3s") == 3723
    assert retry_seconds("no hint") is None


# ---------------- judge ----------------
RECORD = {"question": "Q?", "expected_answer": "E", "answer": "A [ITAct2000.pdf, p. 15, Section 43]",
          "citations": [{"document": "ITAct2000.pdf", "page": 15, "section": "Section 43"}],
          "retrieved_context": CHUNKS}


def test_parse_judgement_clamps_and_handles_garbage():
    out = parse_judgement('text {"correctness": 5, "faithfulness": -2, "relevance": "2", "reason": "ok"} text')
    assert out["correctness"] == 3 and out["faithfulness"] == 0 and out["relevance"] == 2
    assert out["citation_correctness"] is None and out["reason"] == "ok"
    assert all(parse_judgement("not json")[k] is None for k in SCORE_KEYS)


def test_judge_prompt_is_fixed_and_hides_the_method():
    prompt = build_judge_prompt(RECORD)
    assert "EXPECTED ANSWER" in prompt and "RETRIEVED CONTEXT" in prompt and "Section 43" in prompt
    assert "dense" not in prompt.lower() and "hybrid" not in prompt.lower() and "rerank" not in prompt.lower()


def test_judge_answer_returns_structured_scores():
    llm = FakeLLM('{"correctness": 3, "faithfulness": 3, "relevance": 2, "citation_correctness": 3, "reason": "good"}')
    out = judge_answer(RECORD, llm)
    assert out["correctness"] == 3 and out["relevance"] == 2 and out["reason"] == "good"
    assert len(llm.prompts) == 1
