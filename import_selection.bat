@echo off
rem Imports a chosen list of report ids into their own database:  import_selection.bat [--dry-run] [--ids FILE] [--n 100] [--force]
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.import_selection %*
