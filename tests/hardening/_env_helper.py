"""Loads `Env` from tests/sync/conftest.py by file path (never via a
bare `import conftest`/`conftest.py` module of our own) so this package adds
zero risk of colliding with the pre-existing `from conftest import ...` bare
imports used by tests/server's own test modules — pytest's
directory-scoped conftest resolution is sensitive to multiple
identically-named `conftest.py` files existing across the test tree when
plain `import conftest` statements are involved elsewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BIN_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

_GRAPH_SYNC_TESTS_DIR = Path(__file__).resolve().parents[1] / "sync"
FIXTURES_DIR = _GRAPH_SYNC_TESTS_DIR / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"
GRAPHS_DIR = FIXTURES_DIR / "graphs"

_spec = importlib.util.spec_from_file_location(
    "graph_sync_conftest_impl", _GRAPH_SYNC_TESTS_DIR / "conftest.py"
)
_graph_sync_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_graph_sync_conftest)
Env = _graph_sync_conftest.Env
