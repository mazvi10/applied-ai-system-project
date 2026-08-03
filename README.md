# PawPal+ — Pet-Care Scheduler with a Grounded "Ask PawPal" Assistant

## Original project (Modules 1–3)

My original project is **PawPal+**, a Streamlit pet-care planner built on a pure-Python
scheduler (`pawpal_system.py`). It lets an owner register their pets and care tasks
(walks, feeding, meds, enrichment, grooming), then generates a prioritized daily plan
that fits the time available, explains why each task was included or skipped, detects
scheduling conflicts, and auto-recurs daily/weekly tasks. The core classes are `Owner`,
`Pet`, `Task`, and `Scheduler`, with tests covering sorting, filtering, recurrence, and
conflict detection.

## Title and summary

**PawPal+** now adds **Ask PawPal**, a Retrieval-Augmented Generation (RAG) assistant that
answers pet-care questions from a small, hand-vetted knowledge base instead of a language
model's memory. Every answer is grounded in trusted care notes, shows its **sources**, and
is aware of the current owner's pets and pending tasks. This matters because generic
chatbots confidently invent pet-care advice — and bad advice about toxic foods or
medication can hurt an animal. Ask PawPal only answers from vetted notes, cites them, and
pushes urgent or toxic-exposure questions to a vet.

## Architecture overview

The system diagram lives at [`assets/rag_system.mmd`](assets/rag_system.mmd) (Mermaid —
preview it in your IDE). It shows three main components and the data flow between them:

- **Input** — the owner types a question in the Streamlit UI, and their pets + pending
  tasks are pulled in as context.
- **Retriever** (`KnowledgeBase` in `rag.py`) — loads the `knowledge/*.md` notes, splits
  them into paragraph "passages", and ranks them for the question with **BM25 keyword
  search** (`rank_bm25`). Retrieval is deterministic and needs **no API key**, so it is
  fully unit-testable. Passages with a zero relevance score are dropped.
- **Agent** (`ask_pawpal`) — if retrieval finds nothing relevant, it returns a safe
  "not in my notes, ask your vet" answer **without calling the model** (it cannot
  hallucinate). Otherwise it sends the retrieved passages as numbered, titled sources to
  **Gemini (`gemini-3.5-flash`)** with a strict system prompt, and asks for a structured
  JSON answer whose citations quote those sources verbatim.
- **Output** — the answer text plus a **Sources** expander are rendered in the UI.

**Where humans and testing check the AI:** an automated `pytest` suite verifies the
retriever offline; the zero-score gate + no-API fallback is an automated guard against
off-topic hallucination; and the citations let the **owner verify each claim against its
source**, with urgent/toxic issues escalated to a **vet** (the human/professional in the
loop). These checkpoints are color-coded in the diagram.

## Setup instructions

```bash
# 1. From the project directory, create and activate a virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide a Gemini API key (generation only — retrieval and tests need no key).
#    Create a .env file in this directory:
echo "GEMINI_API_KEY=your-key-here" > .env

# 4. Run the app
streamlit run app.py

# 5. (Optional) Run the tests — no API key required
pytest
```

In the app: add an owner, add a pet and some tasks, generate a schedule, then scroll to
the **Ask PawPal** section and ask a question.

## Sample interactions

These are **verbatim outputs** captured from the running system (via `rag.ask_pawpal`).

**1. Toxic-food question — routes to the toxic-foods notes and escalates to a vet**

```text
Q: Can my dog have chocolate?

A: No, your dog should not have chocolate. Chocolate is dangerous for pets because it
   contains theobromine, which dogs cannot process well. Dark and baking chocolate are
   the most toxic, and even small amounts can cause serious symptoms such as vomiting,
   tremors, a racing heart, or seizures. If your pet has consumed chocolate, please
   contact your vet or an animal poison control line immediately.

Sources:
  - [Toxic Foods] "Chocolate contains theobromine, which dogs and cats cannot process
    well; dark and baking chocolate are the most toxic, and even small amounts can cause
    vomiting, tremors, a racing heart, or seizures."
```

**2. Personalized exercise question — uses the owner's pet and pending task**

```text
Owner: Sarah, with puppy "Biscuit" (dog, age 0), pending task: Morning walk
Q: How much should I exercise my new puppy?

A: Hi Sarah! For your new puppy, a helpful guideline is the "five minutes per month of
   age" rule. This means you should aim for about five minutes of structured exercise
   per month of age, once or twice a day. This guideline is designed to protect your
   puppy's growing joints from overexertion while still helping them burn off energy.
   Since you have a "Morning walk" on your pending tasks for Biscuit, keeping this time
   limit in mind will help keep them safe and healthy!

Sources:
  - [Exercise] "For puppies, a common guideline is the 'five minutes per month of age'
    rule: about five minutes of structured exercise per month of age, once or twice a day."
  - [Exercise] "This protects growing joints from overexertion while still burning energy."
```

**3. Off-topic question — no relevant notes, so the safe fallback fires with no model call**

```text
Q: How do I file my taxes?

A: I don't have anything in my notes about that, so I'd rather not guess. For anything
   specific to your pet, please ask your vet.

Sources: (none)
```

## Design decisions

- **Keyword retrieval (BM25), not embeddings.** The corpus is five short docs, so a
  keyword index is accurate, instant, dependency-light, and — crucially — **deterministic
  and key-free**, which makes retrieval unit-testable. Trade-off: BM25 misses pure semantic
  matches, so I added a small stopword list and a tiny stemmer (`bathe/bathing → bath`,
  `vaccinated/vaccinations → vaccin`) to bridge common wording gaps.
- **Grounding over recall.** Passages that score zero are dropped, and if nothing survives,
  the app answers from a fixed fallback **without calling the model**. This can occasionally
  refuse a question the notes actually cover, but it makes hallucination structurally
  impossible on off-topic input — the right trade-off for pet-care advice.
- **Retrieval / generation split.** All the model-free logic lives in `KnowledgeBase`; only
  `ask_pawpal` touches the API, imported lazily. Tests and the whole retrieval path run with
  no key and no network.
- **Structured, cited output.** Gemini has no native citation blocks, so I inline the
  passages as numbered sources and require a JSON response (`response_schema`) with a
  `quote` + `source` per claim. Trade-off: citations are model-produced rather than
  server-verified spans, but the schema keeps parsing reliable and the quotes checkable.
- **Owner-awareness by duck typing.** `ask_pawpal` only needs `.name` and `.pet_list`, so it
  stays decoupled from the scheduler and easy to test with stubs.

## Testing summary

- **What worked:** `tests/test_rag.py` covers the RAG layer with no API key: a chocolate
  question retrieves `toxic_foods.md`, an off-topic question returns no passages, and — the
  key safety test — when context is missing, `ask_pawpal` returns the fixed fallback
  **without ever calling the model** (verified by monkeypatching the API client to raise if
  touched). All **18 tests pass** (15 original scheduler tests + 3 RAG tests). A manual
  smoke test across eight sample questions routes each to the correct document.
- **What didn't work at first:** Raw BM25 sent almost every "how/should/my …" question to
  `feeding.md`, because common function words dominated the score. Vocabulary mismatches
  (`bathe` vs `bath`, `vaccinated` vs `vaccination`) also misrouted questions.
- **What I learned / fixed:** Dropping stopwords and adding a small suffix stemmer fixed the
  routing — a reminder that most RAG quality problems are retrieval problems, not model
  problems, and that a tiny amount of text normalization goes a long way. The generation
  path depends on a live Gemini key, so it is validated manually rather than in CI.

## Reflection

Building Ask PawPal taught me that the hard part of an AI feature is usually **everything
around the model**: what you retrieve, how you constrain it, and how you prove it's right.
The most reliable safety mechanism I added wasn't a clever prompt — it was the deterministic
"return nothing, answer from a fallback" path that never reaches the model at all. Framing
the problem as "retrieve, then ground" made the system both testable and trustworthy.

> The graded responsible-AI reflection — how I collaborated with AI, one helpful and one
> flawed AI suggestion, and the system's limitations — is in
> [`model_card.md`](model_card.md).
