"""Tests for the Ask PawPal RAG retrieval layer.

These exercise ``KnowledgeBase`` and the no-context branch of ``ask_pawpal``
only — retrieval is deterministic BM25 keyword search, and the missing-context
path returns before any API call, so no API key is needed.
"""

import rag
from rag import KnowledgeBase


def test_chocolate_question_retrieves_toxic_foods_doc():
    """A chocolate question should surface the toxic-foods notes first."""
    kb = KnowledgeBase()

    passages = kb.retrieve("Is chocolate safe for my dog?", k=4)

    assert passages, "expected at least one relevant passage"
    # The top hit should come from the toxic-foods document.
    assert passages[0].source == "toxic_foods.md"
    assert "chocolate" in passages[0].text.lower()


def test_offtopic_question_returns_no_passages():
    """A question with no keyword overlap drops all zero-score passages."""
    kb = KnowledgeBase()

    passages = kb.retrieve("quantum chromodynamics lattice gauge theory")

    assert passages == []


def test_ask_pawpal_missing_context_returns_fallback_without_api():
    """When no passage is relevant, ask_pawpal must return the safe fallback
    WITHOUT calling the model — the core no-hallucination guarantee.

    We monkeypatch the module-level API import to explode if it's ever touched,
    proving the missing-context branch returns before any generation call. No
    API key is needed for this test.
    """
    kb = KnowledgeBase()

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("ask_pawpal called the model with no context")

    # ``ask_pawpal`` builds the client via ``genai.Client`` on the generation
    # path only; swap in a landmine so any call would fail loudly.
    import google.genai as genai
    original = genai.Client
    genai.Client = _boom
    try:
        answer = rag.ask_pawpal("how do I file my taxes", knowledge_base=kb)
    finally:
        genai.Client = original

    assert answer.text == rag.NO_MATCH_TEXT
    assert answer.citations == []
    assert answer.retrieved == []
