# Portfolio Q&A Agent — RAG + Conversational Memory

A LangGraph ReAct agent that answers questions about my own projects, resume, and certifications by retrieving relevant facts from a self-built vector store, then remembers the conversation across turns. Built as Week 4 of a self-paced agentic AI curriculum, extending a Week 3 multi-tool weather agent.

**Author:** Archit Bankey · [GitHub: ARCHIT2152](https://github.com/ARCHIT2152) · bankeyarchit52@gmail.com

---

## What it does

Ask it things like:

- *"What was your YOLOv8 model's precision?"*
- *"Which dataset did you use?"* (as a follow-up — it resolves "you" using conversation memory)
- *"What's the weather in Pune?"*
- *"How was the Nvidia internship?"* → correctly says no such internship exists, while surfacing the genuinely related fact that an RTX 4050 GPU was used for training

It has three tools — weather lookup, temperature conversion, and portfolio search — and answers general questions directly without invoking any tool when none is needed.

## Architecture

```
                    ┌─────────────────────┐
                    │   User question      │
                    └──────────┬───────────┘
                               ▼
                 ┌─────────────────────────┐
                 │  LangGraph ReAct Agent   │
                 │  (create_react_agent)    │
                 │  + InMemorySaver         │
                 │    checkpointer          │
                 └──────────┬───────────────┘
                             │ reasons, picks a tool (or none)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
 ┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
 │ weather_tool │     │  convertor    │     │  search_portfolio │
 │ OpenWeather  │     │  °C ↔ °F      │     │  (RAG retrieval)  │
 │     API      │     │               │     │                   │
 └─────────────┘     └───────────────┘     └─────────┬─────────┘
                                                        │
                                            ┌───────────▼────────────┐
                                            │  embedded_chunks.json   │
                                            │  99 chunks, each with   │
                                            │  {source, text,         │
                                            │   embedding}             │
                                            └─────────────────────────┘
```

### RAG ingestion pipeline (one-time, offline)

```
resume.pdf ──► extract.py ──► resume_extracted.txt
                                      │
                                      ▼
                              mark_headers.py ──► resume_marked.txt
                                      │
   3× README.md ────────────────────►│
                                      ▼
                              chunk_text.py ──► chunks.json (~99 chunks)
                                      │
                                      ▼
                             embed_chunks.py ──► embedded_chunks.json
                              (Gemini gemini-embedding-001,
                               task_type=RETRIEVAL_DOCUMENT)
```

At query time, `retrieve.py`'s `search_portfolio` tool embeds the incoming question with `task_type=RETRIEVAL_QUERY`, scores it against all 99 stored vectors with cosine similarity, and returns the top-k matching chunks as plain text for the agent to reason over.

## Tech stack

- **Agent framework:** LangGraph (`create_react_agent`) with LangChain tool decorators
- **LLM:** Gemini (`gemini-3.5-flash`) via `langchain_google_genai`
- **Embeddings:** Gemini `gemini-embedding-001` via the raw `google-genai` SDK
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **PDF extraction:** `pdfplumber`
- **Memory:** LangGraph `InMemorySaver` checkpointer, keyed by `thread_id`
- **Vector search:** hand-written cosine similarity (no external vector DB — see *Design decisions* below)

## Project structure

```
ragbot/data/
├── extract.py              # PDF → raw text (pdfplumber)
├── mark_headers.py         # flags ALL-CAPS lines as section headers
├── chunk_text.py           # splits resume + READMEs into chunks
├── embed_chunks.py         # embeds every chunk via Gemini
├── retrieve.py             # search_portfolio tool (embed query, cosine search)
├── sanity_check.py         # validates embedding quality via cosine similarity
├── chunks.json              # Phase B output
├── embedded_chunks.json    # Phase C output (chunks + vectors)
└── *.md                    # source READMEs (weapon detection, mental health, energy)

lang-graph weather/
└── weather_tool1.py        # the agent: tools, model, checkpointer, chat loop
```

## Setup

```bash
pip install pdfplumber langchain-text-splitters google-genai python-dotenv numpy
pip install langgraph langchain-google-genai langchain-core requests
```

Create a `.env` file (in each project folder, or one shared location) with:

```
GEMINI_API_KEY=your_key_here
WEATHER_API_KEY=your_openweathermap_key_here
```

Run:

```bash
cd "lang-graph weather"
python weather_tool1.py
```

## Build phases

| Phase | What it covers | Status |
|---|---|---|
| A — Data prep | Extract resume PDF, gather READMEs, mark headers | Done |
| B — Chunking | Split text into retrievable chunks | Done |
| C — Embeddings | Convert chunks to vectors via Gemini | Done |
| D — Vector store | Hand-written cosine similarity search | Done |
| E — Tool integration | Wrap retrieval as an agent tool | Done |
| F — Testing | Trigger / skip / honesty test cases | Done |
| G — Memory | Multi-turn conversation via checkpointer | Done |
| H — Wrap-up | Bug fixes, docs, GitHub push | Done |

### Phase A — Data preparation

The resume PDF's project section uses a table layout (`Overview` / `Workflow & Optimization` / `Outcome` as row labels beside content). Neither pdfplumber's default extraction nor `layout=True` reconstructed it as clean prose — one interleaved labels mid-sentence, the other left them as disconnected floating lines. **Decision:** source project content from the three GitHub READMEs instead, which are already clean markdown. The resume extraction is used only for skills, education, certifications, and contact info. A separate script (`mark_headers.py`) detects the resume's ALL-CAPS section headers with `.isupper()` and wraps them in a `##HEADER##` marker, giving the resume the same structural signal the READMEs already have via markdown `##` headers.

### Phase B — Chunking

`RecursiveCharacterTextSplitter` with a separator priority list (`##HEADER##` → markdown `##` → blank lines → line breaks → space) so headers and structure guide splits rather than blind character counts. First pass (`chunk_size=400`) cut sentences and table rows mid-way; increased to `700` and fixed a sequencing bug where a page-marker cleanup regex ran *after* the text had already been split (so it had no effect) — reordering it before the split resolved it.

### Phase C — Embeddings

Chose `gemini-embedding-001` over the newer multimodal `gemini-embedding-2` since the dataset is text-only. Every stored chunk is embedded with `task_type=RETRIEVAL_DOCUMENT`; queries later use `RETRIEVAL_QUERY` — using the wrong one doesn't error, it just silently degrades retrieval accuracy.

**Validating the embeddings took five iterations before the test itself was trustworthy:**

| Attempt | Comparison | Result | What went wrong |
|---|---|---|---|
| 1 | Title vs. Table of Contents | 0.71 (lower than "unrelated" pair) | Neither chunk had real topical content |
| 2 | Two different project titles | 0.77 | Titles share structural similarity regardless of topic |
| 3 | "epochs" keyword match vs. mental health | 0.73 vs. 0.72 | "Epochs" is generic ML vocabulary, not domain-specific |
| 4 | "knife/pistol/CCTV" vs. "inference/OpenCV" | 1.00 | Both filters matched the *same* chunk |
| 5 (final) | "knife/pistol" vs. "surveillance/alert/confidence threshold" | 0.88 vs. 0.73 | Two distinct, topic-specific chunks — clean separation |

Every misleading result traced back to test design (empty content, generic shared vocabulary, or comparing a chunk to itself) rather than the embedding model — a useful reminder that a failed check should prompt investigating the check itself, not just assuming the system under test is broken.

### Phase D — Retrieval

`search_portfolio(query, top_k)` embeds the query, scores it against all 99 stored vectors with the same cosine similarity function validated in Phase C, and returns the top-k chunks. Works reliably on specific technical questions (0.75-0.85+ similarity, correct source every time). Broad questions like *"what projects have you worked on?"* retrieve less cleanly — an accepted, structural limitation of small-chunk RAG rather than a bug: no single chunk contains a summary of all three projects, since chunks were built for factual precision, not synthesis.

### Phase E — Tool integration

`search_portfolio` wrapped with LangChain's `@tool` decorator; its docstring is what the agent's reasoning step reads to decide when to call it. Since the agent (`lang-graph weather/`) and the RAG pipeline (`ragbot/data/`) live in separate project folders, `retrieve.py` uses an absolute file path for `embedded_chunks.json`, and `weather_tool1.py` appends the RAG folder to `sys.path` before importing.

### Phase F — Testing

- Portfolio question → correctly calls `search_portfolio`, answers with real retrieved numbers
- Weather question → still correctly calls `weather_tool`, unaffected by the third tool
- Unrelated question (`24*7`) → answers directly, calls no tool
- Fabricated question ("how was the Nvidia internship?") → correctly states no such internship exists, and offers the genuinely related fact instead (RTX 4050 used for training) rather than hallucinating

### Phase G — Memory

`InMemorySaver` checkpointer passed into `create_react_agent`, with a `thread_id` reused across the whole session. Confirmed working: asking *"What was the precision on the weapon detection project?"* followed by the bare, context-dependent *"which dataset did we use?"* correctly resolved to the same project without it being named again.

### Phase H — Wrap-up

Fixed two lingering bugs in `convertor` from the original weather agent: `from_unit.lower` -> `from_unit.lower()` (missing parentheses meant the method itself, not its result, was being compared, so unit conversion never actually matched), and a casing mismatch (`"Celsius"`/`"fahrenhiet"` vs. the lowercase output of `.lower()`, plus a misspelling of "fahrenheit").

## Design decisions worth calling out

- **Sourced project content from READMEs, not the resume PDF**, after diagnosing that the resume's table layout doesn't survive text extraction in any tested mode — a data-quality problem caught by manually reading extraction output, not by any error or exception.
- **Hand-written cosine similarity instead of a vector database** (Chroma, FAISS, etc.). At ~99 chunks, brute-force comparison is instant, and building it by hand meant fully understanding the mechanism before ever trusting a library's abstraction of it.
- **Both embedding calls (ingestion and query) use the same model and correct, opposite `task_type` values** — a subtle correctness requirement that doesn't throw an error if violated, only silently worse retrieval, so it's called out explicitly here for future reference.

## Known limitations

- Retrieval quality drops on broad/summary-style questions, since chunks are sized for single-fact precision, not multi-project synthesis.
- `InMemorySaver` only persists conversation state in RAM for the current run — memory does not survive restarting the script.
- No quantitative retrieval evaluation (precision@k on a labeled test set) yet — validation so far is qualitative, via the cosine-similarity sanity check and manual query testing.

## Possible next steps

- Add a small labeled question set and measure retrieval precision/recall quantitatively.
- Swap `InMemorySaver` for a persistent checkpointer (SQLite-backed) so memory survives restarts.
- Consolidate the two project folders (`ragbot/data` and `lang-graph weather`) into one repository.
