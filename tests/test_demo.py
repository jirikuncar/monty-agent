"""The README demo end to end, without rendering it: the TUI replays every recorded turn of `traces.json`."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('textual')

import tui

DEMO = Path(__file__).parent.parent / 'docs' / 'assets'
SPEED = 0.2
pytestmark = [pytest.mark.anyio, pytest.mark.slow]


@pytest.fixture
def anyio_backend():
    return 'asyncio'


async def test_demo_replays_every_turn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tui, 'McpQuery', tui.McpQuery)  # the demo replaces it
    monkeypatch.syspath_prepend(str(DEMO))
    monkeypatch.delitem(sys.modules, 'demo', raising=False)
    import demo

    monkeypatch.setattr(demo, 'FPS', 2)
    faster = {'sleep': lambda seconds: asyncio.sleep(seconds * SPEED)}
    monkeypatch.setattr(demo, 'asyncio', SimpleNamespace(**(vars(asyncio) | faster)))
    recorder = await demo.record()
    app = recorder.app
    data = json.loads((DEMO / 'traces.json').read_text())
    assert set(app.nodes) == {n['node'] for n in data['nodes']}
    assert not app.runs and app.selected is not None
    assert 'Code&#160;runs&#160;·' in recorder.frames[-1][1]  # the panel the demo ends on
