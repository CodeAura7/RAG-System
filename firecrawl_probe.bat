@echo off
rem Firecrawl feasibility test on 3-5 PUBLIC report pages (a dry run unless you add --run):  firecrawl_probe.bat 783392 135946 174470 [--run]
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.firecrawl_probe %*
