# Firecrawl for the HackerOne RAG project: research, architecture, Stage 1 test tool, limits and costs

Written 2026-11-05. **Status: Stage 1 only.** One standalone feasibility tool exists and is unit-tested. It has **not** been run against
Firecrawl or HackerOne by the author (no key on the server, and the request would reach hackerone.com). Stage 2 (a real source module) is
designed below but **not built**; it needs the Stage 1 result and a separate approval.

## 1. What Firecrawl is and what it offers (docs read 2026-11-05)

| Item | Finding |
|---|---|
| Scrape | One URL to clean Markdown (or HTML, JSON, screenshot). Runs JavaScript, so single-page apps render. |
| Batch Scrape | A fixed list of URLs in one job: `batch_scrape()` waits, `start_batch_scrape()` + `get_batch_scrape_status()` poll. `maxConcurrency` caps parallel scrapes. **Results are kept 24 hours**, so save them at once. |
| Python | `pip install firecrawl`; `from firecrawl import Firecrawl`; key from `FIRECRAWL_API_KEY`. Version checked here: 4.55.0 (the real `scrape` signature was inspected). Errors: `FirecrawlError`, `RateLimitError`, `PaymentRequiredError`, `UnauthorizedError`, `WebsiteNotSupportedError`, ... |
| Cost | **1 credit per page**. JSON-extraction format +4. Pages that return 403/404 still cost 1. Cached results still cost 1. |
| Plans | Free $0 = 1,000 credits/month, 2 concurrent, 10 scrapes/min. Hobby $19 = 5,000. Standard $99 = 100,000. |
| Proxies | `basic`, `enhanced`, `auto`. **`auto` (the default) retries with enhanced proxies when basic fails.** |
| Not documented | CAPTCHA handling; whether a single `scrape` call checks robots.txt (third-party pages say `crawl` does; not confirmed by Firecrawl); the price of enhanced proxies. |

## 2. HackerOne's terms and robots.txt (checked 2026-11-05)

* `https://hackerone.com/robots.txt` is **one line**: `Sitemap: https://hackerone.com/sitemap.xml`. No Disallow rules.
* Pages read: customer terms (`/terms`), general terms (`/terms/general`), Community Member terms (`/terms/finder`), and the disclosure guidelines, all
  effective 11 May 2026. **None contains a clause about scraping, crawling or bots by a website visitor**, and none addresses copying or reusing
  disclosed report text. The nearest clause binds account holders: "bypass any measures HackerOne may use to prevent or restrict access".
  HackerOne states it claims no ownership of researcher submissions.
* **Not read:** the Code of Conduct, the Copyright and IP policy, each program's own policy, and any separate website terms of use.
  The page contents were summarised by a smaller model, so quotes were not checked word for word.
* **Conclusion:** nothing found forbids a small fetch of public disclosed pages, and nothing grants permission. This is a judgment for the project
  owner and supervisor. HackerOne's sanctioned route is its API (`hrag h1 probe`, already built, never run).
  The project owner chose, on 2026-11-05, to proceed with a test of 3-5 public pages.

## 3. The Hugging Face dataset as the baseline (measured from the cached parquet files)

* 12,618 reports, created 2011-07-27 to 2024-05-07.
* **3,136 (24.9%) have an empty body and every one of them is marked `visibility = no-content`.** HackerOne itself shows no body for them,
  so a page scrape is not expected to recover them (inference; one test page would confirm it).
* Of a random 1,500 non-empty reports (project's own `clean_report` + `assess`): steps present in 48%, a PoC in 39%, an impact statement in 66%,
  flagged `sparse` 27%, flagged `incomplete` 36%. Median cleaned length 1,220 characters.
* The dataset has **no comments column**. A page would add the discussion and status changes, plus anything newer than May 2024.
  Scrape and Batch Scrape fetch URLs they are given. Finding the ids of newer reports would need Firecrawl's separate map / crawl / search
  features (bulk crawling of hackerone.com, not planned here) or another source such as the HackerOne API.

## 4. Architecture

```
 existing                                        new (Stage 2, NOT built)
 ---------------------------------------------   -------------------------------------------
 hrag h1 fetch      -> data/cache/h1/*.json      
 hrag gitlab fetch  -> data/cache/gitlab/*.json  hrag firecrawl fetch -> data/cache/firecrawl/*.json
          \                    |                          /
           +----> collect.load_sources  (small hook: read the new cache folder too)
                         |
                  SourceReport(source="firecrawl", source_url=..., body=<page body markdown>)
                         |
        clean_report -> assess -> one report per id -> near-duplicate removal -> select      (all REUSED)
                         |
              enrich -> chunk (c256/c400/c600) -> embed (openai/text-embedding-3-large, 3072) -> SQLite   (all REUSED)
```

* **Reused unchanged:** cleaning and prompt-injection removal, quality flags, one-report-per-id (better quality wins, a tie goes to the API),
  duplicate removal, selection caps, enrichment, chunking, embedding, storage, retrieval.
* **New in Stage 2:** (a) a fetch-and-cache client modelled on `gitlab_src.py`; (b) **cutting the report body out of the page's menus, headers
  and sidebars**. This is the real work and cannot be designed until Stage 1 shows the page layout; (c) a roughly 5-line hook in `collect.load_sources`.
* **Databases:** Stage 2 would write to its own database through its own config file (the same pattern as `config.selection.toml`).
  `hrag.db` and `hrag.selection.db` are never opened for writing.
* **Keys:** `FIRECRAWL_API_KEY` is read with the project's existing lookup (environment, then `.env`, then the OS credential store).

## 5. Stage 1: what was built (all new files; nothing existing is changed)

| File | Purpose |
|---|---|
| `tools/firecrawl_probe.py` | The feasibility tool |
| `tests/test_firecrawl_probe.py` | 56 tests, no network |
| `firecrawl_probe.bat` | Windows launcher (sets `PYTHONPATH`, uses `.venv`) |
| `requirements-firecrawl.txt` | Adds the `firecrawl` package for this tool only |

Rules the tool enforces in code (each has a test, and each was deliberately broken once to prove the tests notice):

| Rule | How |
|---|---|
| Public report pages only | only `https://hackerone.com/reports/<number>`; look-alike hosts, `user@host` tricks, query strings, other paths refused |
| Tiny | at most 5 reports per run; there is no flag to raise it |
| No bypassing | `proxy="basic"` (never `auto`), no `actions`, headers, location or mobile options |
| Polite | one request per page, at least 6 s apart, **no retries** |
| Stops at the first problem | blocked status (401/403/429/451/503), challenge or login page, empty page, 404, any exception |
| Dry run by default | real requests need `--run` and a key |
| No overwrite, free re-runs | a cached page is reused and never replaced; only successful pages are cached |
| Privacy | the page is not kept in Firecrawl's cache (`store_in_cache=False`); output and `summary.json` hold numbers and short headings only, never report text or the key; error messages have the key removed |

What it measures per report: whether the dataset's text appears on the page (share of 5-word groups found), page length versus dataset length,
whether the page has steps / PoC / impact the dataset lacks, signs of comments or status changes, and a heading outline (to design Stage 2).

## 6. Test results (2026-11-05)

| Check | Result |
|---|---|
| New tests | 56 passed |
| Whole project suite, with the JSON fix applied | **307 passed** (251 existing + 56 new) |
| Deliberate breakage of 11 rules (wrong proxy, no stop after a block, 6 reports, key not removed, any host, no spacing, status ignored, cached page re-requested, dry run sending, browser actions added, summary leaking text) | all 11 caught |
| Real Firecrawl SDK 4.55.0 pointed at a local fake server | accepted every argument; the request on the wire has `proxy: basic`, `formats: [markdown]`, `storeInCache: false`, no actions; the key is only in the Authorization header |
| **Not run** | Firecrawl's real service; any hackerone.com report page; extraction quality; the page layout |

## 7. Limitations

* Completeness of extraction is **unknown** until Stage 1 is run. Report pages are probably rendered by JavaScript; this was not checked.
* The no-content reports (24.9% of the dataset) are probably unrecoverable.
* Attachments and screenshots appear as links only. The existing `image_dependent` flag would apply.
* Discovery of 2025/2026 reports is a separate problem: Scrape needs the URLs first (see section 3).
* The probe needs the dataset parquet files (in `data/cache`) for the comparison; without them it still reports page metrics.
* The Hugging Face dataset text stays private: nothing here publishes it, and summaries contain no report text.

## 8. Costs

| Scenario | Credits | Plan needed |
|---|---|---|
| Stage 1 test, 3 pages | about 3 | Free (1,000/month) |
| Stage 1 test, 5 pages | about 5 | Free |
| 100 reports (one full collection) | about 100 | Free |
| All 9,482 non-empty dataset reports (not planned) | about 9,500 | Standard $99 (Hobby's 5,000 is too few) |

Failed pages (403/404) still cost 1. Free plan: 10 scrapes a minute, so 5 pages at the 7-second default take about 30 seconds.
Embedding the resulting text with OpenRouter is a separate cost and is not estimated here.

## 9. Sources

* Firecrawl: [Scrape](https://docs.firecrawl.dev/features/scrape), [Batch Scrape](https://docs.firecrawl.dev/features/batch-scrape),
  [Proxies](https://docs.firecrawl.dev/features/proxies), [Python SDK](https://docs.firecrawl.dev/sdks/python), [Pricing](https://www.firecrawl.dev/pricing)
* HackerOne: [robots.txt](https://hackerone.com/robots.txt), [Terms](https://www.hackerone.com/terms), [General terms](https://www.hackerone.com/terms/general),
  [Community Member terms](https://www.hackerone.com/terms/finder), [Disclosure guidelines](https://www.hackerone.com/disclosure-guidelines)
