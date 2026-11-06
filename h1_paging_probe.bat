@echo off
rem How does the official HackerOne listing really paginate (page-size cap, next link, page numbers, headers)? Structure only, <= 10 requests.
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.h1_paging_probe %*
