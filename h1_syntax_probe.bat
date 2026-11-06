@echo off
rem Which listing queries does the API really obey (exact times, upper bounds, extra filters, sort keys, page numbers)? About 25 requests.
rem   h1_syntax_probe.bat --manifest data\h1_discovery\<time>\manifest.json
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.h1_syntax_probe %*
