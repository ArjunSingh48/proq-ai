"""Vercel serverless entry point — wraps the FastAPI app."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so we can import app modules
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app import app  # noqa: E402  — the FastAPI instance
