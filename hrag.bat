@echo off
rem Runs the hackerone-rag CLI from anywhere:  hrag.bat <command> ...
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m hrag %*
