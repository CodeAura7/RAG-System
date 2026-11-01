import sys

# Windows consoles often default to a legacy code page; reports contain arbitrary Unicode.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from .cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
