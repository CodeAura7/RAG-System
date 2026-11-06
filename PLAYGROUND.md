# RAG Playground

A local web page for testing the RAG system you already have. It runs on your laptop only (bound to 127.0.0.1), uses your
existing `hrag.db`, `.env` and `config.toml`, and calls the project's own retrieval, embedding and answer code. Nothing is simulated.

## What is in this update

New files only. **No existing file was modified.**

| File | What it is |
|---|---|
| `playground/app.py` | the Streamlit page (3 tabs: Ask, Chunk Explorer, Evaluation) |
| `playground/backend.py` | the logic behind it; reuses `hrag.retrieve`, `hrag.answer`, `hrag.evaluate`, `hrag.cli.make_retriever` |
| `playground/__init__.py` | makes `playground` importable |
| `tests/test_playground_backend.py` | 19 automated tests for the backend (fake embedder / fake LLM, no network) |
| `playground.bat` | starts the page |
| `requirements-playground.txt` | the project's requirements + `streamlit` |
| `PLAYGROUND.md` | this file |

## Install (once)

Unzip over your project folder. Windows asks about overwriting nothing, because none of these files exist yet.
Your `.env`, `data\`, `eval\`, `.venv\` and `config.toml` are not in the ZIP and are not touched.

```powershell
.venv\Scripts\python -m pip install -r requirements-playground.txt
```

## Run

```powershell
.\playground.bat
```

A browser tab opens at http://127.0.0.1:8501. Stop it with Ctrl+C in the terminal.

## What you need in the database first

The page reads what the command line already built:

```powershell
.\hrag.bat stats        # reports, chunks per set, embeddings
.\hrag.bat build        # if chunks / embeddings are missing
```

* **Vector and hybrid search** need every chunk of the chosen chunk set embedded with the chosen embedding profile
  (default `openrouter_3_large`, unchanged). If not, the page says how many are missing and offers BM25 only.
* **Other chunk sets** (`c256`, `c600`, `whole`) show up in the dropdown when they exist. To compare them with vectors:
  `.\hrag.bat chunk --set c256` then `.\hrag.bat embed --set c256` (costs a few cents).

## Answers need an answer model

Retrieval works with only your OpenRouter key (it embeds the question). To **generate answers** the page needs a chat model with a key:

* the default is `claude_sonnet` and needs `ANTHROPIC_API_KEY` in `.env`; or
* use OpenRouter for answers too. `openrouter_enrich` is **too small** for this (`max_tokens = 300` cuts the JSON answer off).
  Add this block to the end of `config.toml` (your edit, your choice of model: pick any chat model id from openrouter.ai/models):

```toml
[llm.openrouter_answer]
provider = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
model = "openai/gpt-4o-mini"      # example only; choose the model you want to test
api_key_name = "OPENROUTER_API_KEY"
max_tokens = 1500
temperature = 0.0
```

The sidebar lists every profile and marks the ones without a key. For the optional judge, choose a model from a different
company than the answer model.

## How to use it

**Ask tab**
1. Type a question. *Retrieve only* costs one embedding call. *Retrieve + answer* also calls the answer model.
2. The answer cites sources as [1], [2]. (The model itself is asked, as before, to cite `[C123]` chunk ids; the page only renumbers
   them for reading. A citation of a chunk the model was never shown is flagged as invented.)
3. Below the answer: the top **K** chunks (what the model saw) and, separately, the chunks ranked K+1 to the judging depth that
   retrieval found but the model never saw. Each shows chunk id, report id, title, program, weakness, section, final score,
   vector rank and BM25 rank, the arithmetic behind the fused score, and the complete chunk text.
4. Mark each chunk *relevant* / *irrelevant* (judge the chunk, not the report). Rate the answer. *Save evaluation*.

**Chunk Explorer tab** - filter and pick any report, see its full text, every chunk with id, section and estimated tokens, the text
on either side of each chunk boundary, and warnings (split code fences, chunks starting mid-sentence, very small chunks).
The bottom table compares all chunk sets on these heuristics.

**Evaluation tab** - Precision@K, Recall@K, MRR, nDCG@K, latency and your answer ratings, per configuration and per saved
evaluation (CSV download). *Compare* re-runs your saved questions under other chunk sets and modes and scores them against the
reports you marked relevant.

## Reading the numbers honestly

* The metrics are computed from **your marks**. Unjudged chunks count as not relevant, so look at `judged@K` before trusting `P@K`.
* **Recall is relative**: relevant chunks in the top K ÷ all chunks you marked relevant for that question in the list you were shown.
  The true number of relevant chunks in the corpus is unknown.
* Ten questions give noisy numbers; a gap of 0.05 between two configurations is not a finding.
* Chunk size comparisons across chunk sets are made at **report level**, because chunk ids differ between sets.
* Token counts are the project's estimate (characters / 3.6), not an exact tokenizer count.
* The page shows the top K that the answer model receives. Chunks removed before ranking as near-duplicates, or beyond
  2 per report (`max_chunks_per_report`), are not listed anywhere; that is how the project's retriever works.

## Where things are saved

`eval\playground\judgments.jsonl` - one JSON line per saved evaluation. The database is never written. Delete the file to start over.

## Privacy / safety

Runs only on 127.0.0.1; usage statistics are switched off in `playground.bat`. API keys are read exactly as the command line reads them
and are never displayed. Report text and model answers are untrusted: chunk text is shown as plain text, and markdown images in
answers are neutralised so a report cannot make your browser fetch an address.

## One thing to know about reuse

Two small underscore-prefixed methods of the retriever (`_vector_ranked`, `_bm25_ranked`) are called to read the per-leg scores that
`Retriever.search` computes but does not return. A test checks their ranks against the ranks `search` itself reports. If you ever
change `hrag/retrieve.py`, run `.venv\Scripts\python -m pytest tests\test_playground_backend.py`.
