# AI Interactions Log

---

## Agent Workflow (SF7)

I asked the agent to add a grounded "Ask PawPal" RAG assistant to my existing
PawPal+ scheduler. My prompting focused on two things:

1. **How to implement RAG *safely*.** I prompted for a design that answers only
   from a trusted knowledge base (not the model's memory), cites its sources,
   and refuses to guess. I specifically asked that retrieval be deterministic
   and key-free so it could be unit-tested, that off-topic questions return a
   "not in my notes, ask your vet" fallback *without* calling the model (no
   hallucination), and that urgent/toxic questions be pushed to a vet.

2. **How to wire the new system into the app.** I then prompted for help
   connecting the RAG module to the Streamlit UI and the rest of the project —
   reusing the already-resolved `owner`, caching the knowledge base, showing a
   "Sources" expander, and later swapping the LLM backend to Gemini and loading
   the key from a `.env`.

**What did the agent do?**

- Built `knowledge/` (5 vetted markdown notes), `rag.py` (BM25 retriever +
  grounded `ask_pawpal`), tests, and updated `requirements.txt`.
- Wired it into `app.py`: `import rag`, an `@st.cache_resource` knowledge base,
  and an "Ask PawPal" section reusing `owner`, rendering the answer + sources.
- Produced the system diagram (`assets/rag_system.mmd`), rewrote the `README`,
  and ran the tests / retrieval smoke tests at each step.

**What did you have to verify or fix manually?**

- I switched the generation backend from the agent's original Anthropic/Claude
  implementation to **Google Gemini** (`gemini-3.5-flash`) with structured-JSON
  citations and a `.env` key; the agent then re-synced the README and diagram to
  match the actual code.
- I asked it to confirm the safety guarantee with a real test: it added one
  proving `ask_pawpal` returns the fallback **without ever calling the model**
  when context is missing (18 tests pass).
- Capturing live Gemini output for the README samples is still pending, so those
  answers remain labeled "representative."


