@echo off
rem What do a few disclosed reports' relationships (generated content, activities, attachments, summaries) hold? Names, counts and lengths only.
rem   h1_relations.bat                      (uses the 3 sample ids that "hrag.bat h1 probe" saved)
rem   h1_relations.bat 4152553 4029867      (up to 5 ids)
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.h1_relations %*
