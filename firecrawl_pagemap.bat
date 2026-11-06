@echo off
rem OFFLINE profile of pages firecrawl_probe already cached (no credits, no requests):  firecrawl_pagemap.bat 783392 135946 174470
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m tools.firecrawl_pagemap %*
