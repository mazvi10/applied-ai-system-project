# Model Card — Ask PawPal (RAG assistant)

## What it is

Ask PawPal is a Retrieval-Augmented Generation feature added to the PawPal+
pet-care app. It answers pet-care questions using a small, hand-vetted knowledge
base (`knowledge/*.md`) rather than the model's own memory: BM25 keyword search
retrieves the most relevant passages, and Gemini (`gemini-3.5-flash`) writes a
short, cited answer grounded strictly in those passages plus the owner's pets and
pending tasks. If nothing relevant is retrieved, it returns a fixed "ask your vet"
message without calling the model at all.

## How I used AI during development

I used an AI agent to design and build the feature end to end: scaffolding the
knowledge base, writing the BM25 retriever and the grounded `ask_pawpal`
function, wiring it into the Streamlit UI, generating the architecture diagram,
and writing the tests. I steered it with two main prompts: one asking how to
implement RAG *safely* (grounded, cited, no hallucination), and one asking for
help wiring the new module into the existing app. I reviewed and tested each
change rather than accepting it blindly, and I made the final call on the model
provider and the safety behavior.

## One helpful AI suggestion

Splitting **retrieval** from **generation**, and having off-topic questions fall
back to a fixed answer *without calling the model*. This made the anti-
hallucination guarantee structural rather than prompt-based: retrieval is
deterministic and needs no API key, so it's fully unit-testable, and a question
with no relevant notes can never reach the model to make something up.

## One flawed AI suggestion

The AI's first retriever tokenized text without removing stopwords, so common
words like "how", "should", and "my" dominated the BM25 score — nearly every
question ("how often should I walk my puppy?", "when do kittens get vaccinated?")
routed to `feeding.md`. I caught this with a retrieval smoke test and fixed it by
adding a small stopword list and a tiny stemmer (`bathe`/`bathing` → `bath`),
after which all sample questions routed to the correct document. (The AI also
first built the feature on Anthropic/Claude; I switched it to Gemini to match my
setup.)

## System limitations

- **Citations are model-produced.** Gemini returns the quotes; they aren't
  verified against the source text, so a misquote is possible.
- **Keyword retrieval only.** BM25 can miss heavily paraphrased questions that
  share no keywords with the notes.
- **Tiny knowledge base.** Five documents cover common topics; anything outside
  them hits the safe fallback rather than being answered.
- **Not veterinary advice.** Answers are general and defer urgent/toxic issues to
  a vet or poison-control line.

## Future improvements

- Add hybrid retrieval so paraphrased questions still match.
- Expand and version the knowledge base, with review before changes ship.
- Mock the Gemini client to cover the generation path in automated tests.
