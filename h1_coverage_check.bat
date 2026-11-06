@echo off
rem Audit a discovery manifest against the API, month by month (listing endpoint only; counts, dates and report ids).
rem   h1_coverage_check.bat --manifest data\h1_discovery\<time>\manifest.json
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.h1_coverage_check %*
