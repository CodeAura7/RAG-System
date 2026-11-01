# hackerone-rag

A small, local RAG (retrieval-augmented generation) system over about 100 **publicly disclosed HackerOne reports**,
built to answer pentesting questions with citations. Everything runs on your own computer: the reports, the SQLite database
and the code. Only report text (to the embedding / LLM provider) and questions (to the answer model) leave your machine.
It is a standalone project and does not touch Bhedan or Smriti.

```
 HackerOne API ─┐                                            ┌─ clean ─ flag ─ dedupe ─ select ─┐
  (your token)  ├─► fetch (cached on disk) ─► collect ───────┤                                  ├─► SQLite
 GitLab copies ─┘     h1 fetch / gitlab fetch   (offline)    └─ pinned sample links always in ──┘   reports
                                                                                                     │
      question ─► semantic (vector) + keyword (BM25) ─► merged by rank (hybrid) ─► answer + citations  ◄── embed ◄── chunk ◄── enrich ("When to try this")
```

## What you need

* Windows 10/11 and **Python 3.11 or newer** (python.org installer, tick "Add python.exe to PATH"). VS Code is optional.
* Keys, all kept **outside the code** in a local `.env` file (see Setup):

| Key | Needed for |
|---|---|
| `H1_API_USERNAME`, `H1_API_TOKEN` | the HackerOne API collector (`h1 probe / discover / fetch`) |
| `OPENROUTER_API_KEY` | embeddings (`openai/text-embedding-3-large`, 3072 dimensions) and the "When to try this" LLM |
| `GITLAB_TOKEN` *(optional)* | GitLab comments only. Everything else on GitLab works without any account |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` *(optional)* | only `ask` and `eval answers` (the answer model and the judge, see `config.toml`) |

Free accounts with no payment method are not enough for the embedding step. The whole 100-report corpus is only tens of
thousands of tokens, so embedding costs a fraction of a cent.

## Setup (once)

1. Unzip the folder anywhere and open it in VS Code (`File → Open Folder`).
2. In the VS Code terminal run `.\setup.bat` (creates `.venv` and installs numpy, httpx, pyarrow, keyring).
3. VS Code → `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `.venv`.
4. Copy `.env.example` to `.env` and fill in your keys. `.env` is git-ignored and never put in the ZIP. Lookup order for every key:
   a real environment variable → `.env` → Windows Credential Manager (`.\hrag.bat keys set NAME`).
5. `.\hrag.bat doctor` shows where each key was found (`env`, `.env`, `keyring` or `missing`) - never its value.

`.\hrag.bat <command>` is the same as `.venv\Scripts\python -m hrag <command>`.

## Use it, step by step

### 1. Test your HackerOne token first (nothing else is allowed to run before this passes)
```
.\hrag.bat h1 probe
```
About 5 read-only requests. It answers one question: **does the API return the complete report text (reproduction steps,
impact) or only summaries?** It prints HTTP codes, field *names* and text *lengths* - never your token or report text - and saves
`data\h1_probe.json`. `h1 discover` and `h1 fetch` refuse to run until that file says the API is usable.
If it says "NOT usable", nothing was collected; send me the printed output (it is safe to share).

### 2. Collect reports from the HackerOne API
```
.\hrag.bat h1 discover --n 94 --per-program 3     # list public disclosed reports across different programs
.\hrag.bat h1 fetch                               # download them (cached in data\cache\h1, polite request spacing)
```

### 3. Add the GitLab copies
Put your sample links (one per line) in `sources\gitlab_urls.txt`, then:
```
.\hrag.bat gitlab fetch                           # your sample links: always included in the collection
.\hrag.bat gitlab discover --n 30                 # OPTIONAL top-up from GitLab's public HackerOne-labelled issues
```
GitLab's copies are all one program (GitLab), so top-up reports are capped (`max_gitlab`, default 30) and marked
`source = gitlab`. Without `GITLAB_TOKEN`, GitLab comments are skipped (GitLab answers 401), and the run says so.

### 4. Build the database
```
.\hrag.bat collect          # offline and free: clean, flag, deduplicate, select ~100 -> SQLite   (prints a full summary)
.\hrag.bat enrich           # writes "When to try this" for every report (LLM; a template if there is no key)
.\hrag.bat chunk            # section-based chunks with a header and metadata on each
.\hrag.bat embed            # embeddings via OpenRouter
```
or just `.\hrag.bat build` (collect if the database is empty → enrich → chunk → embed). Each step is safe to re-run.
`collect` refuses to replace an existing database without `--force`, because that deletes chunks and embeddings (paid work).

### 5. Ask questions
```
.\hrag.bat stats
.\hrag.bat search "SSRF through a webhook URL" --show-text
.\hrag.bat ask "Where should I test for an OAuth redirect token leak?"
```
`search` shows the chunks, their score, which search found them (`via {'vector': 3, 'bm25': 1}` = rank in each) and report metadata. Choose the search with `--mode vector|bm25|hybrid` (default `hybrid`); `--mode bm25` needs no API key at all. `ask` answers only from the retrieved chunks, as JSON with
citations that are checked in code, and may abstain. Citations link to the report's real source (HackerOne or GitLab).

### Commands

| Command | What it does |
|---|---|
| `doctor` / `keys` | environment check; which keys are found and where |
| `h1 probe / discover / fetch` | HackerOne API collector (gated on a passing probe) |
| `gitlab fetch / discover` | your sample links / optional top-up from GitLab's public copies |
| `collect [--n --max-gitlab --force]` | clean + flag + dedupe + select → SQLite; writes `data\collect_summary.json` |
| `enrich [--llm --force --require-llm]` | the "When to try this" text |
| `chunk [--set c400]` / `embed [--set --embed --force]` | chunks / embeddings (resumable, saved every 128 chunks) |
| `build` | collect (if empty) → enrich → chunk → embed |
| `search QUERY` / `ask QUESTION` | `--mode vector\|bm25\|hybrid`, `--k 5`, `--rerank`, `--set`, `--embed` |
| `eval draft / validate / retrieval / answers / compare` | evaluation, see below |
| `ingest-dataset` | **legacy, optional**: the old Hugging Face dataset path (a fallback if the API proves unusable) |

## What the cleaning does, and why

Each report goes through `hrag/clean.py` before anything is stored. Every rule has a test, and the rules were checked against
real public GitLab copies, which is how two template variants and a false positive were found.

* **GitLab boilerplate** is removed: the `HackerOne report #N … by user` header line and contents line, the attachment list, the
  empty "How To Reproduce" template, related-issue links, the internal implementation plan, environment-info dumps, quick actions
  (`/label …`) and empty template headings. Real content under those headings is kept; the template's own headings are renamed
  ("What is the current bug behavior?" → "Current behavior"). Substantive human comments (triage verdicts, root cause, fix) are
  kept under "Triage discussion"; thanks/status chatter, bots and system notes are not.
* **Instructions addressed to AI tools are removed** from the prose (prompt-injection defence): "ignore previous instructions",
  "note to the AI assistant", "mark this report as critical", role-marker tokens, hidden HTML comments, and invisible /
  zero-width / tag characters. The same phrases **inside code blocks or in quotes are kept** - a report about prompt
  injection needs its payload as PoC - but they are counted and flagged `ai_instructions_in_code`. A title that carries an
  instruction is replaced. Report text is also always sent to an LLM inside `<report>` tags marked as untrusted data.
* **Screenshots, GIFs and videos** are replaced by `[image: name]` / `[video: name]` and recorded in the database. Nothing is
  downloaded or described (flag-only). A report whose steps depend on them is flagged `image_dependent`.
* **Placeholders and secrets** are normalised: `Cookie:` / `Authorization:` / CSRF header values, JWTs, `glpat-…`-style tokens,
  secret-looking URL parameters and non-example emails become `<COOKIE>`, `<TOKEN>`, `<CSRF_TOKEN>`, `<JWT>`, `<EMAIL>`;
  `YOUR_TOKEN`, `{{token}}`, `<your token>`, `[REDACTED]` and `█████` become `<TOKEN>` / `<REDACTED>`. **Endpoints, parameter
  names and payloads are not touched** (`/api/v4/projects/:id/releases`, `tag_name`, `<script>`, `${IFS}`, XML elements).
* HTML entities (`&#34;`, `&lt;`) are decoded for GitLab copies only. Cleaning is idempotent (cleaning twice changes nothing).

### Quality flags and selection
Flags stored on every report **and on every chunk**: `sparse`, `no_steps`, `no_poc`, `no_impact`, `incomplete` (no steps and
no PoC), `disputed` (outcome, labels or comments say not-applicable / informative / by design / cannot reproduce),
`has_images`, `image_dependent`, `ai_instructions_removed`, `ai_instructions_in_code`. A 0-1 score (steps 25%, PoC 25%,
impact 20%, length 15%, not disputed 15%, minus penalties) is used only to prefer better reports.

`collect` keeps up to `n_reports`: your pinned sample links first, then strong reports, then flagged ones only if the quota is
still open (they stay flagged). Caps keep it diverse: `per_program` (3), `per_weakness` (8), `max_gitlab` (30). One report per id
(better quality wins, ties go to the API), then exact and near-duplicate removal. If fewer than requested are available the summary says
**SHORTFALL** instead of hiding it.

### Chunking, header, metadata, "When to try this"
* Reports under the size budget stay whole. Longer ones are split **at section boundaries**, never inside a code block; a step
  that introduces a request/command/payload ("2. Send this request:") is glued to the code block below it so a boundary can't
  separate them. Size, overlap and strategy are configurable per chunk set in `config.toml` (`c256`, `c400`, `c600`, `whole`).
* Every chunk starts with a descriptive **header** (report id, title, program, weakness, severity, outcome, section) and a line
  `When to try this: …`, and that header is embedded with the text. Every chunk also stores a **metadata snapshot** of its report
  (program, weakness/CWE, severity, source and URL, flags, score, images, section) in the database.
* **"When to try this"** is 1-2 generic sentences per report saying under which conditions a pentester should try the technique
  (feature, technology, signal). The LLM sees the report as untrusted data; any answer that is empty, too short or looks like an
  instruction is discarded for a deterministic template.

### Search - three modes
* **`vector`** (semantic): exact cosine similarity between the question and the stored embeddings (NumPy; at this size no index is needed). Finds
  paraphrases: "make the server fetch an internal URL" ≈ SSRF.
* **`bm25`** (keyword): an in-memory BM25 index over the chunk texts (header included). It keeps security identifiers whole *and* in parts -
  `CVE-2021-44228` is searchable as itself and as `44228`; `/api/v1/users` as itself and as `api`, `users` - so exact names, parameters and
  CVEs are found even when the embedding misses them. Needs no embeddings, so it works without any API key.
* **`hybrid`** (default): both lists merged with Reciprocal Rank Fusion, score = Σ 1 / (60 + rank) (Cormack et al., SIGIR 2009). Scores of
  different kinds cannot be added, so it fuses by rank. A chunk both searches like ranks above one only a single search found.

Optional reranking (`--rerank`) reorders the candidates with a hosted reranker. At most 2 chunks per report are returned and a chunk almost
identical to a better one is dropped. SQLite FTS5 is deliberately not used: Python's bundled SQLite doesn't guarantee it on every Windows build.

## Configuration (`config.toml`, no secrets in it)

`[collect]` sizes and caps · `[chunk_sets.*]` chunk size / overlap / minimum · `[embeddings.*]` profiles (default
`openrouter_3_large` = `openai/text-embedding-3-large`, 3072 dimensions, via OpenRouter) · `[llm.*]` profiles (`openrouter_enrich` for
"When to try this", `claude_sonnet` for answers, a judge) · `[retrieval]` (`mode` = vector | bm25 | hybrid, `rrf_k`, `top_k`, caps) · `[ingest]` (legacy dataset path).

## Evaluation (kept from the original project)

1. **Questions.** Write about half yourself and let an AI draft the rest - drafts are never trusted:
   `.\hrag.bat eval draft --n 40 --unanswerable 10` writes `eval\questions.draft.jsonl` with every item `needs_review=true`.
   Fix `gold_report_ids` and `key_facts`, set `needs_review` to `false`, add your own, save as `eval\questions.jsonl`
   (format: `eval\questions.example.jsonl`). The evaluation commands **refuse** to run while any question is unreviewed.
   `eval validate` checks the file and that every gold report exists.
2. **Retrieval:** `.\hrag.bat eval retrieval --arms vector,bm25,hybrid,hybrid+rerank` → Recall@1/3/5, MRR, nDCG@10, duplicate rate, latency,
   plus a paired bootstrap against the first arm. Per-question CSV/JSON go to `eval\results\`.
3. **Answers and citations:** `.\hrag.bat eval answers` → abstention, citation validity and precision (checked in code), key-fact
   coverage and faithfulness (judge model - use a *different company* than the answer model; the command warns otherwise).
4. **Compare embedding models or chunk sizes** on your own questions: `embed --embed <profile>`, `chunk --set c256`, then
   `eval retrieval` and `eval compare`. With ~60 questions one question is ~1.7 points: treat small gaps as ties and write your pass marks down first.

## Honest limits - what has and has not been verified

* **Verified:** 170 automated tests (no network, no keys, fake local servers); the cleaner and the GitLab path were run against
  real public GitLab issues; the whole chain `gitlab discover → fetch → collect → enrich(template) → chunk` ran against the live
  GitLab API.
* **NOT verified:**
  * **The HackerOne API.** The JSON field names are assumptions taken from the documentation. `h1 probe` prints the real ones;
    if they differ, only `hrag/h1_api.py` (`to_source_report`, `report_id_of`) needs adjusting. It may also turn out that your
    token can only read summaries - the probe tells you before anything is collected.
  * **OpenRouter:** that it accepts the `dimensions` parameter for `openai/text-embedding-3-large` (if not, set `dimensions = 0`)
    and that `openai/gpt-4o-mini` is the id it lists today (change `[llm.openrouter_enrich] model` otherwise).
  * **Windows:** built and tested on Linux (Python 3.12). `setup.bat` and `hrag.bat` have not been run on Windows.
  * Embedding quality on security text, and whether reranking helps at ~300 chunks, are unmeasured.
  * The AI-instruction filter is a pattern list, not a guarantee. It reduces risk; treat all report text as untrusted.
* **Data terms.** HackerOne API text and GitLab's copies are subject to their owners' terms. Keep the data private (`data/` is
  excluded from the ZIP and git), do not redistribute it, and confirm with whoever owns the policy that sending it to an
  embedding / LLM provider is allowed.
* Token counts are estimates (characters ÷ 3.6).

## Tests
```
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

## Layout
```
config.toml           all settings (no secrets)           hrag/clean.py      cleaning, injection filter, placeholders, images
.env.example          copy to .env for your keys          hrag/quality.py    flags + score
sources/gitlab_urls.txt  your sample links                hrag/collect.py    clean → dedupe → select
hrag/h1_api.py        HackerOne API collector + probe     hrag/enrich.py     "When to try this"
hrag/gitlab_src.py    GitLab copies                       hrag/chunking.py   section-based chunker, headers
hrag/store.py         SQLite (auto-upgrades old files)    hrag/retrieve.py   vector search, rerank, dedupe
hrag/embeddings.py hrag/llm.py hrag/rerank.py             hrag/evaluate.py   metrics, answers, citations
hrag/secrets.py       env / .env / credential store       hrag/cli.py        commands
```
