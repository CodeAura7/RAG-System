# Import a chosen list of reports (friend's 90 + 10 top-up = 100)

`ingest-dataset` can only pick reports at random (a seed). This tool takes a **list of report ids you chose**, keeps all of
them, and tops up to 100. It writes to its **own** database, so your existing `data\hrag.db` (the 30 GitLab reports) is never touched.

It adds files only. Nothing in your project is modified, and nothing is sent to any paid API.

## What it does

1. Reads the ids in `sources\selection_ids_friend90.txt` (the 90 from your friend's CSV; ids only, no text).
2. Takes those reports' text from the Hugging Face dataset you already download (`data\cache`; it downloads ~19 MB the first time).
3. Runs them through the **same** cleaning as `python -m hrag collect`: HTML/boilerplate cleanup, removal of text aimed at an AI
   reader (prompt injection), quality flags, exact and near-duplicate removal.
4. Keeps **every** id you gave (the per-program and per-weakness caps do not apply to them). If one is dropped (too short after
   cleaning, or a duplicate of another), it is listed with the reason. It never disappears silently.
5. Tops up to `--n` (100) from the **300 most-voted other** reports. Inside that pool the project's selector prefers stronger
   reports (no weak flags, higher quality score), then the lowest report id, and applies the usual caps. **These top-ups are
   chosen by this tool, not by your friend**; the run prints them under "NOT from your list".
6. Writes `data\hrag.selection.db`, `data\hrag.selection.summary.json`, and `config.selection.toml` (a copy of your
   `config.toml` where only `[paths] db` and `eval_dir` differ, so evaluation files go to `eval_selection\`).

## Run it (PowerShell, in the project folder)

```powershell
.\import_selection.bat --dry-run      # shows what would be selected; writes nothing
.\import_selection.bat                # writes the database + config.selection.toml
.\hrag.bat --config config.selection.toml build      # enrich + chunk + embed (needs your OpenRouter key)
.\hrag.bat --config config.selection.toml stats
.\hrag.bat --config config.selection.toml search "ssrf cloud metadata credentials" --mode bm25
```

Playground: start `.\playground.bat` and put `config.selection.toml` in the "config path" box in the sidebar.

Running it a second time stops and asks for `--force` (replacing the selection database deletes its chunks and embeddings, which
cost money to rebuild). It also refuses outright if `--out-db` is the database named in your config.

## Things to know

- **A hand-picked list is easier than reality.** All 90 are `resolved` reports with steps, impact and a fix section. That is good for a
  first retrieval-accuracy test, but do not quote the numbers as if they applied to all HackerOne reports.
- **Two of your friend's reports have almost the same title** (ids 1735586 and 1784999, same program, "Wordpress users
  Disclosure"). Their texts are only about 52% similar (the duplicate filter's cut-off is 80%), so both are kept. For questions
  about that issue, either may be the right answer, which can lower precision. Judge them both relevant in the Playground.
- **`build` costs a little.** It calls your enrich model (template text without a key) and embeds about 350 chunks with
  `openai/text-embedding-3-large`. One OpenRouter key covers both. I have not priced it.
- **Dataset terms:** the text is from `Hacker0x01/hackerone_disclosed_reports`, "made available under HackerOne's Terms of
  Service". Reuse terms beyond that were not found: keep it for your own private use, do not redistribute the database.
- Use a different list with `--ids path\to\ids.txt` (one id per line, `#` comments allowed) or a `.csv` with an `id` column.
- `--n 120` asks for a different total; `--pool-top-votes 600` widens the top-up pool.
