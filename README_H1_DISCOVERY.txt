HACKERONE API DIAGNOSTICS - version 4 (no more date leaps, busy days reported, built-in audit, syntax probe)
==========================================================================================================
Read-only tools. They use ONLY the official HackerOne API (api.hackerone.com) with your own token from the .env file next to config.toml.
No existing project file is changed, neither SQLite database is opened, no report page on hackerone.com is requested, no attachment is
downloaded, no full report content is saved, nothing goes to Firecrawl. Never paste the token into chat.

WHAT WENT WRONG IN VERSION 3 (your coverage check found it: 805 reports, 168 of 449 sampled missing, mostly 2026)
  When a single day holds more reports than one page shows (about 50), version 3 fell back to page numbers for that day and then moved its
  date marker to the newest day on those pages. Page numbers on this API return the wrong records, so the marker LEAPT over months of
  reports and nothing noticed. My statement that the default "never uses page numbers" was wrong for exactly those busy days.

WHAT CHANGED IN VERSION 4
  * h1_discover: a busy day can no longer move the marker by more than ONE day. Page numbers are used on a busy day only to COLLECT extra
    reports, never to move the marker. Each busy day is recorded ("gap_days" in manifest.json, "BUSY DAY" lines) and the run is then NOT
    called complete: it says "N busy days may be partial".
  * h1_discover now ends with its own month-by-month audit (about 50 extra requests; --no-audit to skip). It only prints "complete" when
    that audit finds no gap. The audit samples the first reports of each month, so it cannot see holes INSIDE a busy day - those are what
    gap_days lists.
  * NEW h1_syntax_probe: finds out which queries the API really obeys, judged by the returned items: exact timestamps in six spellings,
    upper date bounds, extra filters (substate, severity_rating, team, program) that could split a busy day, other sort keys, and what
    page[number] really does (checked against the order of your own 2025 reports, which passed the coverage check).

1. Unzip. These files REPLACE the same-named files of earlier diagnostics zips (and add the syntax probe), so -Force is needed.
   The zip contains only the 16 files listed at the end - nothing else in your project is touched:
     Expand-Archive -Path "$env:USERPROFILE\Downloads\hackerone-rag-h1-diagnostics-v4.zip" -DestinationPath . -Force

2. Syntax probe on your 805-report manifest (about 25 requests, 0.6 seconds apart). This decides how busy days can be listed completely:
     .\h1_syntax_probe.bat --manifest data\h1_discovery\<the 805-report folder>\manifest.json
   Read the "what this means for busy days" lines at the end. If the manifest path is wrong the message says so; list manifests with:
     Get-ChildItem data\h1_discovery -Recurse -Filter manifest.json | Select-Object FullName,Length,LastWriteTime

3. New discovery (about 60 to 150 requests including the audit):
     .\h1_discover.bat
   Expected now: far more than 805 reports, lines "BUSY DAY <date>" for the mass-disclosure days, "final audit ... " output, and an honest
   "INCOMPLETE ... busy days may be partial" if any busy day exists. That is correct behaviour, not an error.

4. Send Claude (attach): data\h1_syntax\<time>\syntax.json, the new manifest.json (it contains page_log, gap_days and audit), and the
   printed "what this means for busy days" lines. They hold statuses, counts, dates, parameter names and report ids - no report text.

Tests (optional):  .venv\Scripts\python -m pytest -q tests\test_h1_relations.py tests\test_h1_discover.py tests\test_h1_paging_probe.py tests\test_h1_coverage_check.py tests\test_h1_syntax_probe.py

FILES IN THIS ZIP (16): README_H1_DISCOVERY.txt,
  h1_discover.bat, h1_relations.bat, h1_paging_probe.bat, h1_coverage_check.bat, h1_syntax_probe.bat,
  tools\h1_discover.py, tools\h1_relations.py, tools\h1_paging_probe.py, tools\h1_coverage_check.py, tools\h1_syntax_probe.py,
  tests\test_h1_discover.py, tests\test_h1_relations.py, tests\test_h1_paging_probe.py, tests\test_h1_coverage_check.py, tests\test_h1_syntax_probe.py
To undo: delete those files and the folders data\h1_relations, data\h1_discovery, data\h1_paging, data\h1_coverage, data\h1_syntax.
