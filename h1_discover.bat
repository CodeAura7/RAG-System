@echo off
rem Discover disclosed reports since a date through the OFFICIAL HackerOne API (metadata only) and count them by month, severity and CWE.
rem   h1_discover.bat --max-items 300       (small trial first)
rem   h1_discover.bat                       (everything since 2025-01-01)
rem   h1_discover.bat --since 2025-06-01 --until 2025-06-30
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.h1_discover %*
