"""Pytest plugin: isolate legacy native synthetic checkpoint writes.

No historical assertions are weakened. Modules are collected normally; only
the scratch output root used by checkpoint unit tests is redirected.
"""
import importlib
from pathlib import Path
import tempfile


def pytest_collection_modifyitems(session, config, items):
    native = [item for item in items if 'test_native.py' in str(item.path)]
    if not native:
        return
    root = Path(__file__).resolve().parents[2] / 'data/research_cleanup_local/test_scratch'
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='native_', dir=root))
    common = importlib.import_module('scripts.native_channel_research.common')
    common.LOCAL = destination
    for item in native:
        item.module.LOCAL = destination
