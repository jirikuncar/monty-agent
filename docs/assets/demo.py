"""Record the README demo: the TUI replaying real, anonymized Logfire records (`traces.json`).

    CHROMIUM=/path/to/chrome uv run --extra tui --with playwright docs/assets/demo.py

Logfire and Agent Sandboxes are stood in for: queries are answered from `traces.json`, and each turn replays the
nodes the real one recorded, at a quarter of its pace. Frames are Textual screenshots, rendered with Playwright and
joined with ffmpeg into `demo.gif` and `demo.mp4`.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import tui
from oauth import Login
from trace_store import Node, iso

SIZE = (132, 38)
PACE = 0.25
FPS = 8
AGES = (timedelta(minutes=25), timedelta(hours=2), timedelta(days=1))


class Replay:
    """Logfire as the TUI queries it, holding what has been recorded so far."""

    def __init__(self, data: dict):
        self.data = data
        self.rows: list[dict] = []
        self.costs: dict[str, float] = {}
        self.executions: list[dict] = []
        self.parents = {n['node']: n['parent'] for n in data['nodes']}
        self.turns: dict[str, list[dict]] = {}
        for node in data['nodes']:
            self.turns.setdefault(node['trace'], []).append(node)
        roots = sorted((n for n in data['nodes'] if n['parent'] is None), key=lambda n: n['at'], reverse=True)
        live = next(r for r in roots if r['text'].startswith('Calculate tides'))
        self.live = [t for t, nodes in self.turns.items() if self.root(nodes[0]) == live['node']]
        now = datetime.now(UTC)
        for age, root in zip(AGES, [r for r in roots if r is not live], strict=True):
            for trace, nodes in self.turns.items():
                if self.root(nodes[0]) == root['node']:
                    replayed = {n['at']: now - age + timedelta(seconds=n['at'] - root['at']) for n in nodes}
                    for node in nodes:
                        self.record(node, replayed[node['at']])
                    self.finish(trace, replayed)

    def root(self, node: dict) -> str:
        node_id = node['node']
        while self.parents[node_id] is not None:
            node_id = self.parents[node_id]
        return node_id

    def record(self, node: dict, at: datetime) -> Node:
        fields = {k: v for k, v in node.items() if k not in ('trace', 'span', 'at') and v is not None}
        self.rows.append(
            {
                'trace_id': node['trace'],
                'span_id': node['span'],
                'start_timestamp': at.isoformat(),
                'attributes': json.dumps({f'monty_agent.{k}': v for k, v in fields.items()}),
            }
        )
        attrs = {k: v for k, v in fields.items() if k not in ('node', 'parent', 'kind', 'text')}
        return Node(
            node['node'], node['parent'], node['kind'], node['text'], iso(at), attrs, node['trace'], node['span']
        )

    def finish(self, trace: str, replayed: dict[float, datetime]) -> None:
        """A turn's cost and code runs are queryable once it ends; a run goes right after the step that asked for it."""
        self.costs[trace] = self.data['costs'].get(trace, 0.0)
        for run in self.data['executions']:
            if run['trace'] == trace:
                step = max(at for at in replayed if at <= run['at'])
                at = replayed[step] + timedelta(milliseconds=1)
                self.executions.append(
                    {**run, 'trace_id': trace, 'span_id': run['span'], 'start_timestamp': at.isoformat()}
                )

    async def __call__(self, sql: str, params: dict, since: datetime | None = None) -> list[dict]:
        if 'per_service' in sql:
            return [{'trace_id': t, 'cost': c} for t, c in self.costs.items()]
        if "'child sandbox'" in sql:
            return [r for r in self.executions if r['trace_id'] in sql]
        if 'gen_ai' in sql or "archive' IS NOT NULL" in sql:
            return []
        return [r for r in self.rows if since is None or datetime.fromisoformat(r['start_timestamp']) >= since]

    def set_token(self, token: str) -> None:
        pass

    async def aclose(self) -> None:
        pass


async def replay_turn(app: tui.MontyTui, replay: Replay, trace: str, text: str) -> None:
    """What `MontyTui.run_turn` records, node by node, as the real turn did."""
    app.runs[id(trace)] = tui.Run(text or 're-run')
    nodes = replay.turns[trace]
    replayed, previous = {}, nodes[0]['at']
    for node in nodes:
        await asyncio.sleep(min((node['at'] - previous) * PACE, 2.0))
        previous = node['at']
        replayed[node['at']] = datetime.now(UTC)
        app.add_node(replay.record(node, replayed[node['at']]))
    del app.runs[id(trace)]
    replay.finish(trace, replayed)
    app.costs[trace] = replay.costs[trace]
    app.executions.pop(trace, None)
    app.show_tree()
    app.select(app.selected)


class Recorder:
    def __init__(self, app: tui.MontyTui):
        self.app = app
        self.frames: list[tuple[float, str]] = []
        self.started = time.monotonic()
        self.ended = 0.0

    async def run(self) -> None:
        while True:
            svg = self.app.export_screenshot(title='monty-agent')
            if not self.frames or self.frames[-1][1] != svg:
                self.frames.append((time.monotonic() - self.started, svg))
            self.ended = time.monotonic() - self.started
            await asyncio.sleep(1 / FPS)

    def write(self, gif: Path, mp4: Path) -> None:
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as tmp, sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get('CHROMIUM'))
            page = browser.new_page()
            lines = []
            for index, (at, svg) in enumerate(self.frames):
                page.set_content(f'<body style="margin:0">{svg}</body>')
                png = Path(tmp) / f'{index:05}.png'
                page.locator('svg').first.screenshot(path=png)
                until = self.frames[index + 1][0] if index + 1 < len(self.frames) else self.ended
                lines += [f"file '{png}'", f'duration {until - at:.3f}']
            lines.append(lines[-2])
            browser.close()
            concat = Path(tmp) / 'frames.txt'
            concat.write_text('\n'.join(lines))
            ffmpeg = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat)]
            scale = 'scale=1280:-2:flags=lanczos'
            subprocess.run(
                [*ffmpeg, '-vf', f'{scale},format=yuv420p', '-r', '25', '-movflags', '+faststart', mp4], check=True
            )
            palette = (
                f'{scale},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4'
            )
            subprocess.run([*ffmpeg, '-vf', palette, '-r', '10', gif], check=True)


async def type_text(pilot, text: str) -> None:
    for char in text:
        await pilot.press('space' if char == ' ' else char)
        await asyncio.sleep(0.04)


async def pick(pilot, node_id: str) -> None:
    widget = next(w for w in pilot.app.query(tui.Pickable) if getattr(w.item, 'id', None) == node_id)
    widget.scroll_visible(animate=False, top=True)
    await asyncio.sleep(1)
    await pilot.click(widget)


async def record() -> Recorder:
    data = json.loads((HERE / 'traces.json').read_text())
    replay = Replay(data)
    tui.McpQuery = lambda *args, **kwargs: replay
    args = Namespace(
        remote='https://logfire-eu.pydantic.dev/',
        model='gateway/anthropic:claude-haiku-5-5',
        script=Path('examples/general_agent.py'),
        poll=0,
    )
    login = Login('https://logfire-eu.pydantic.dev', 'demo', time.time() + 86400, 'demo', 'https://example.invalid')
    app = tui.MontyTui(args, login)
    turns = list(replay.live)
    app.run_turn = lambda node, text: app.run_worker(replay_turn(app, replay, turns.pop(0), text), group='turns')
    question = next(n['node'] for n in data['nodes'] if n['kind'] == 'question')
    last_step = next(n['parent'] for n in data['nodes'] if n['kind'] == 'fork')
    answer = next(n['text'] for n in data['nodes'] if n['kind'] == 'answer' and n['text'] != 'tomorrow')

    async with app.run_test(size=SIZE) as pilot:
        recorder = Recorder(app)
        frames = asyncio.create_task(recorder.run())
        await asyncio.sleep(2.5)

        await pilot.press('ctrl+n')
        await type_text(pilot, 'Calculate tides based on the Earth and Moon positions for a given date.')
        await pilot.press('enter')
        await asyncio.sleep(4)

        await type_text(pilot, 'tomorrow')
        await pilot.press('enter')
        await asyncio.sleep(9)

        await pick(pilot, question)  # roll back: answer the question again
        await asyncio.sleep(1.5)
        await type_text(pilot, 'tomorrow')
        await pilot.press('enter')
        await asyncio.sleep(3.5)

        await type_text(pilot, answer)
        await pilot.press('enter')
        await asyncio.sleep(13)

        await pick(pilot, last_step)  # fork: re-run the last model step only
        await asyncio.sleep(1.5)
        await pilot.press('enter')
        await asyncio.sleep(6)

        await pilot.press('ctrl+t')
        await asyncio.sleep(4)
        frames.cancel()
    return recorder


if __name__ == '__main__':
    asyncio.run(record()).write(HERE / 'demo.gif', HERE / 'demo.mp4')
