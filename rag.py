"""Grounded "Ask PawPal" assistant.

A small Retrieval-Augmented Generation (RAG) layer over a trusted, hand-vetted
knowledge base of pet-care notes. Retrieval is deterministic keyword search
(BM25) so it needs no API key and is unit-testable; only answer *generation*
calls the Gemini API, and it is grounded strictly in the retrieved passages
plus the current owner's pets and tasks — never the model's own memory.

Design split (so tests can exercise retrieval offline):
  * ``KnowledgeBase`` loads the docs and ranks passages — pure Python.
  * ``ask_pawpal`` retrieves, then (only if there's something to ground on)
    calls Gemini with the passages inlined as numbered, titled sources and
    asks for a structured JSON answer whose citations quote those sources.

Gemini has no native document-citation blocks like Anthropic's, so grounding is
enforced two ways: the passages are the only source material in the prompt, and
the model must return each cited claim's exact quote and source title as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

# Where the vetted markdown/text notes live, relative to this file.
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# Grounded, non-hallucinating fallback when nothing in the notes is relevant.
NO_MATCH_TEXT = (
    "I don't have anything in my notes about that, so I'd rather not guess. "
    "For anything specific to your pet, please ask your vet."
)

# We honour the caller's explicit model choice; Flash is a good fit for a
# short, grounded, citation-heavy answer and is inexpensive.
DEFAULT_MODEL = "gemini-3.5-flash"

# Kept short so the answer stays focused; citations carry the detail.
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are Ask PawPal, a friendly pet-care assistant inside the PawPal+ app. "
    "Answer ONLY using the provided documents and the owner/pet information in "
    "the user's message. Do not rely on outside knowledge. If the documents "
    "don't cover the question, say so plainly and suggest asking a vet — do "
    "not make things up. Keep answers short and practical, and personalise "
    "them to the owner's pets and pending tasks when it's relevant. You are "
    "not a veterinarian: for anything urgent, or if a pet may have eaten "
    "something toxic, tell the owner to contact their vet or an animal poison "
    "control line immediately."
)


# Function words carry no topic signal but recur across every doc, so left in
# they let a passage rank on "how/should/my" overlap instead of real keywords
# (e.g. a feeding paragraph outscoring the exercise doc on "walk my puppy").
# Dropping them keeps retrieval keyword-driven. Kept small and hand-picked so
# the module stays dependency-free (no NLTK download).
STOPWORDS = frozenset(
    """
    a an and are as at be but by can could do does for from had has have how i
    if in into is it its me my of on or should so than that the their them then
    there these they this to was we were what when where which who will with
    would you your
    """.split()
)


# Conservative suffix rules so morphological variants a user might type match
# the docs' wording: "bathe/bathing/baths" -> "bath", "vaccinated/vaccinations"
# -> "vaccin", "puppies" -> "puppy". Longest suffixes first so "vaccinations"
# strips before the plain "s" rule fires. This is a deliberately tiny stemmer,
# not full Porter — enough to bridge the common gaps without a dependency.
_SUFFIXES = (
    ("ations", ""), ("ation", ""), ("ated", ""), ("ating", ""),
    ("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""),
)


def _stem(word: str) -> str:
    """Strip a common English suffix so word variants share one token."""
    for suffix, replacement in _SUFFIXES:
        # Keep a 3+ char root so short words aren't mangled (e.g. "is").
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-word tokenizer shared by indexing and querying.

    Deliberately simple: BM25 just needs consistent tokens on both sides. We
    drop stopwords so scoring keys on topical words, and light-stem the rest so
    morphological variants (bathe/bathing, vaccinated/vaccination) still match.
    """
    words = "".join(
        ch if ch.isalnum() else " " for ch in text.lower()
    ).split()
    return [_stem(word) for word in words if word and word not in STOPWORDS]


@dataclass
class Passage:
    """One paragraph-sized chunk of a knowledge doc, with its provenance."""

    title: str  # the doc's "# heading", used as the citation label
    source: str  # filename the passage came from, e.g. "toxic_foods.md"
    text: str  # the paragraph itself


@dataclass
class Citation:
    """A grounded claim: text the model wrote, backed by a quoted source."""

    claim: str  # the model's sentence(s) this citation supports
    quote: str  # the exact quoted span from the source document
    source: str  # the document title the quote came from


@dataclass
class Answer:
    """The full result of an Ask PawPal question."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Passage] = field(default_factory=list)


def _split_passages(title: str, source: str, body: str) -> list[Passage]:
    """Split a doc body into blank-line-separated paragraph passages.

    The "# Title" heading line is dropped from the chunks (it's captured as the
    title instead), so each passage is a self-contained paragraph.
    """
    passages: list[Passage] = []
    for block in body.split("\n\n"):
        text = block.strip()
        if not text:
            continue
        # Drop a leading heading line if this block still carries one.
        lines = [ln for ln in text.splitlines() if not ln.startswith("# ")]
        text = "\n".join(lines).strip()
        if text:
            passages.append(Passage(title=title, source=source, text=text))
    return passages


class KnowledgeBase:
    """Loads the pet-care notes and answers retrieval queries with BM25.

    The corpus is tiny and static, so a keyword index built once at construction
    is plenty — no embeddings or vector store. ``retrieve`` is deterministic and
    needs no API key, which is what makes it unit-testable.
    """

    def __init__(self, knowledge_dir: Path | str = KNOWLEDGE_DIR) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.passages: list[Passage] = self._load_passages()
        # A BM25 index over the tokenized passage texts. Guard against an empty
        # corpus so an empty knowledge dir degrades to "no matches" gracefully.
        self._corpus_tokens = [_tokenize(p.text) for p in self.passages]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self.passages else None

    def _load_passages(self) -> list[Passage]:
        """Read every .md/.txt file into title + paragraph passages."""
        passages: list[Passage] = []
        for path in sorted(self.knowledge_dir.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            body = path.read_text(encoding="utf-8")
            # The first "# heading" is the document title; fall back to the
            # filename stem if a doc somehow lacks one.
            title = path.stem.replace("_", " ").title()
            for line in body.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            passages.extend(_split_passages(title, path.name, body))
        return passages

    def retrieve(self, question: str, k: int = 4) -> list[Passage]:
        """Return up to ``k`` passages most relevant to ``question``, best first.

        Zero-score passages (no keyword overlap) are dropped, so an off-topic
        question can return fewer than ``k`` passages — or none at all, which
        lets the caller decline to answer instead of hallucinating.
        """
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(question))
        # Rank passage indices best-first, keeping only positive scores.
        ranked = sorted(
            (i for i, score in enumerate(scores) if score > 0),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [self.passages[i] for i in ranked[:k]]


def _summarize_owner(owner: object | None) -> str:
    """Render a plain-text summary of an owner's pets and pending tasks.

    Duck-typed against ``pawpal_system.Owner``: we only need ``.name`` and
    ``.pet_list``, and treat a task as pending when its status is PENDING. Kept
    tolerant of missing attributes so a partial/stub owner won't crash the call.
    """
    if owner is None:
        return "No owner is signed in, so answer generally."

    # Imported lazily so retrieval/tests don't depend on the scheduler module.
    try:
        from pawpal_system import PENDING
    except Exception:  # pragma: no cover - fallback if run standalone
        PENDING = "pending"

    name = getattr(owner, "name", "the owner")
    pets = getattr(owner, "pet_list", []) or []
    if not pets:
        return f"The owner is {name}, who has no pets on file yet."

    lines = [f"The owner is {name}. Their pets and pending tasks:"]
    for pet in pets:
        species = getattr(pet, "animal_type", "pet")
        age = getattr(pet, "age", "unknown")
        pending = [
            getattr(task, "description", "task")
            for task in getattr(pet, "tasks", [])
            if getattr(task, "status", None) == PENDING
        ]
        pending_text = ", ".join(pending) if pending else "none"
        lines.append(
            f"- {getattr(pet, 'name', 'Unnamed')} "
            f"({species}, age {age}); pending tasks: {pending_text}"
        )
    return "\n".join(lines)


# JSON shape we ask Gemini to return: the grounded answer, plus a list of
# citations that each quote an exact span from one of the numbered sources.
# Gemini enforces this via ``response_schema`` so we can parse without guessing.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["source", "quote"],
            },
        },
    },
    "required": ["answer", "citations"],
}


def _build_content(question: str, owner: object | None,
                   passages: list[Passage]) -> str:
    """Assemble the user prompt: numbered titled sources, then owner + question.

    Gemini has no document-citation blocks, so each passage is inlined as a
    numbered "Source" labelled with its title. The model is told these are the
    only material it may use and that every citation's ``quote`` must be copied
    verbatim from one of them, with ``source`` set to that source's title.
    """
    source_blocks = "\n\n".join(
        f"Source {i} (title: {passage.title}):\n{passage.text}"
        for i, passage in enumerate(passages, start=1)
    )
    return (
        "Use ONLY the sources below to answer. Quote spans verbatim from them "
        "in your citations, and set each citation's `source` to the exact "
        "title shown for the source you quoted.\n\n"
        f"{source_blocks}\n\n"
        f"{_summarize_owner(owner)}\n\n"
        f"Question from the owner: {question}"
    )


def _parse_answer(response: object, retrieved: list[Passage]) -> Answer:
    """Turn Gemini's JSON response into an Answer with text and citations.

    ``response_schema`` guarantees the shape, but we stay defensive: a blocked
    or empty response degrades to an apologetic answer rather than crashing.
    """
    import json

    raw = getattr(response, "text", None)
    if not raw:
        return Answer(
            text="Sorry, I couldn't produce an answer just now. Please try "
            "again.",
            citations=[],
            retrieved=retrieved,
        )

    data = json.loads(raw)
    answer_text = (data.get("answer") or "").strip()
    citations = [
        Citation(
            claim=answer_text,
            quote=(cite.get("quote") or ""),
            source=(cite.get("source") or ""),
        )
        for cite in data.get("citations", [])
    ]
    return Answer(text=answer_text, citations=citations, retrieved=retrieved)


def ask_pawpal(
    question: str,
    owner: object | None = None,
    knowledge_base: KnowledgeBase | None = None,
    k: int = 4,
    model: str = DEFAULT_MODEL,
) -> Answer:
    """Answer a pet-care question, grounded in the knowledge base.

    Retrieves the top-``k`` passages for ``question``. If nothing relevant is
    found, returns a polite "not in my notes" answer WITHOUT calling the API —
    so we never invent facts. Otherwise sends the passages as numbered, titled
    sources plus a summary of the owner's pets and pending tasks, and returns
    Gemini's grounded answer with its citations.
    """
    kb = knowledge_base or KnowledgeBase()
    retrieved = kb.retrieve(question, k=k)

    if not retrieved:
        return Answer(text=NO_MATCH_TEXT, citations=[], retrieved=[])

    # Imported here so that retrieval (and its tests) never require the SDK or
    # an API key to be installed/configured.
    from google import genai
    from google.genai import types

    # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=_build_content(question, owner, retrieved),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
            # Disable "thinking" to match the original's fast, direct answers.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    return _parse_answer(response, retrieved)
