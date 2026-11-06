FIRECRAWL FEASIBILITY TEST - how to add it and run it (about 10 minutes)
=======================================================================

What this is
  A small standalone tool that fetches 3 public HackerOne report pages through Firecrawl, then tells you (numbers only, no report
  text) whether the page contains the whole report and what it has that the Hugging Face data does not. It is a TEST. It does not
  touch any database, config.toml, the JSON-fix files, or any existing cache.

Files in this zip (all NEW; none replaces an existing file)
  tools\firecrawl_probe.py          the tool
  tests\test_firecrawl_probe.py     56 tests, no network
  firecrawl_probe.bat               launcher
  requirements-firecrawl.txt        adds the 'firecrawl' package for this tool only
  FIRECRAWL.md                      research, architecture, results, limits, costs
  README_APPLY_FIRECRAWL.txt        this file

STEPS (PowerShell, in D:\Shiv\rag system)

1. Unzip WITHOUT -Force, so PowerShell refuses to overwrite anything that already exists:
     Expand-Archive -Path "$env:USERPROFILE\Downloads\hackerone-rag-firecrawl-probe.zip" -DestinationPath .
   If it complains that a file exists, stop and tell Claude.

2. Install the one extra package into your project's .venv:
     .venv\Scripts\pip install -r requirements-firecrawl.txt

3. Get a Firecrawl key (free: 1,000 credits a month, no card) at https://www.firecrawl.dev and add ONE line to your .env file:
     FIRECRAWL_API_KEY=fc-xxxxxxxx
   The key goes only in .env on your laptop. Never paste it into chat, code or a zip.

4. Run the tests:
     .venv\Scripts\python -m pytest -q
   Expected: 307 passed (251 existing + 56 new). If you have not pasted the openrouter_answer block into config.toml yet you will
   see 1 extra failure; that is the JSON-fix test, not this tool.

5. DRY RUN first (sends nothing, reads no key):
     .\firecrawl_probe.bat 783392 135946 174470
   These are three reports that are in the Hugging Face data: a detailed SSRF report, a very short one, and a long CSRF report with images.

6. The real run (about 3 Firecrawl credits, under a minute):
     .\firecrawl_probe.bat 783392 135946 174470 --run
   It asks for ONE page at a time, waits 7 seconds between pages, and STOPS at the first block, challenge page, empty page or error.
   If it stops: do NOT retry and do NOT change any setting to get past it. Tell Claude what it printed.

7. Send Claude the file  data\cache\firecrawl_probe\summary.json  (numbers and short headings only, no report text, no key),
   plus the lines the tool printed.

How to read the result
  page_contains_dataset_text   the page holds the whole report the dataset has (good: complete extraction is possible)
  partial / different_text     the page does not match the dataset text (the extraction is incomplete or the layout needs work)
  dataset_has_no_body          the dataset had nothing for that report; the page numbers show whether the page has more
  "page adds: has_poc, ..."    things the page has that the dataset text lacks
  activity markers             how many comment / status-change phrases the page has (the dataset has no comments at all)

Rules the tool follows (so you can tell your supervisor): public report pages only, at most 5 per run, basic proxy only, no browser
actions, one request per page with no retries, stops at the first problem, nothing is stored in Firecrawl's cache, and the key is
never printed or saved.

TO UNDO: delete the 5 new files above, and the folder data\cache\firecrawl_probe if it exists. (Optional: .venv\Scripts\pip uninstall firecrawl)
