@echo off
rem Starts the RAG Playground (local only: bound to 127.0.0.1, no usage statistics sent).
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0playground\app.py" --server.address 127.0.0.1 --browser.gatherUsageStats false %*
