#!/usr/bin/env python
"""Top-level ZeroPawn CLI entrypoint."""

import sys
from pathlib import Path

# Add repo root to path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from python.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
