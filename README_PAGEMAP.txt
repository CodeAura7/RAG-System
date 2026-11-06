FIRECRAWL PAGE MAP - offline, read-only (2 minutes)
===================================================
Profiles the 3 pages that firecrawl_probe already saved. It makes NO request, uses NO Firecrawl credit and needs NO key.
It prints numbers and short section headings only - never report text.

1. Unzip WITHOUT -Force (PowerShell refuses to overwrite anything):
     Expand-Archive -Path "$env:USERPROFILE\Downloads\hackerone-rag-firecrawl-pagemap.zip" -DestinationPath .
2. Run it:
     .\firecrawl_pagemap.bat 783392 135946 174470
3. Paste Claude the printed lines (and, if you like, data\cache\firecrawl_probe\pagemap.json - numbers and headings only).
4. Optional tests: .venv\Scripts\python -m pytest -q     (expect 313 passed on your laptop if test_report_library.py is absent, else 326)

To undo: delete firecrawl_pagemap.bat, tools\firecrawl_pagemap.py, tests\test_firecrawl_pagemap.py and data\cache\firecrawl_probe\pagemap.json
