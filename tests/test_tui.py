"""The TUI against an in-memory stand-in for Logfire: no login, sandbox or network."""

import json
import time
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('textual')

import tui  # after the skip, so a run without the `tui` extra skips these tests
from oauth import Login

pytestmark = pytest.mark.anyio
T1, T2 = 'a' * 32, 'b' * 32


@pytest.fixture
def anyio_backend():
    return 'asyncio'


def row(trace: str, at: str, node: str, parent: str | None, kind: str, text: str = '', **attrs: Any) -> dict:
    fields = {'node': node, 'parent': parent, 'kind': kind, 'text': text, **attrs}
    attributes = {f'monty_agent.{k}': v for k, v in fields.items() if v is not None}
    timestamp = f'2026-11-18T{at}Z'
    return {
        'trace_id': trace,
        'span_id': f'span-{node}',
        'start_timestamp': timestamp,
        'attributes': json.dumps(attributes),
    }


class FakeLogfire:
    """Answers the TUI's queries from lists, honoring `since` the way the query window does."""

    def __init__(self, nodes: list[dict]):
        self.nodes = nodes
        self.costs: dict[str, float] = {}
        self.executions: list[dict] = []
        self.archive_events: list[dict] = []
        self.calls: list[tuple[str, datetime | None]] = []

    async def __call__(self, sql: str, params: dict, since: datetime | None = None) -> list[dict]:
        kind = 'cost' if 'per_service' in sql else 'code' if 'child sandbox' in sql else 'other'
        kind = 'requests' if 'gen_ai' in sql else 'archive' if "archive' IS NOT NULL" in sql else kind
        kind = 'nodes' if kind == 'other' else kind
        self.calls.append((kind, since))
        if kind == 'cost':
            return [{'trace_id': t, 'cost': c} for t, c in self.costs.items() if 'IN (' not in sql or t in sql]
        rows = {'code': self.executions, 'requests': [], 'archive': self.archive_events, 'nodes': self.nodes}[kind]
        return [r for r in rows if since is None or datetime.fromisoformat(r['start_timestamp']) >= since]

    def set_token(self, token: str) -> None:
        pass

    async def aclose(self) -> None:
        pass


def make_app(monkeypatch: pytest.MonkeyPatch, logfire: FakeLogfire, **options: Any) -> tuple[tui.MontyTui, list]:
    monkeypatch.setattr(tui, 'McpQuery', lambda *args, **kwargs: logfire)
    args = Namespace(
        remote='https://logfire-us.pydantic.dev/', model='test', script=Path('examples/general_agent.py'), poll=0
    )
    for key, value in options.items():
        setattr(args, key, value)
    login = Login('https://logfire-us.pydantic.dev', 'token', time.time() + 3600, 'client', 'https://e/token', 'r')
    app = tui.MontyTui(args, login)
    launched: list[tuple[str | None, str]] = []
    app.run_turn = lambda node, text: launched.append((node and node.id, text))  # no sandbox
    return app, launched


CONVERSATION = [
    row(T1, '10:00:00', 'p1', None, 'prompt', 'Fibonacci 30?', model='m-a'),
    row(T1, '10:00:01', 'm1', 'p1', 'model', messages=1),
    row(T1, '10:00:05', 'm2', 'm1', 'model', messages=3),
    row(T1, '10:00:09', 'o1', 'm2', 'output', '832040'),
    row(T2, '10:05:00', 'p2', 'o1', 'prompt', 'and 31?', model='m-a'),
    row(T2, '10:05:01', 'e2', 'p2', 'error', 'MontyRefusedError: unavailable'),
]
OTHER = [row('c' * 32, '09:00:00', 'q1', None, 'prompt', 'other conversation')]


async def test_collapsed_conversation_shows_its_last_message(monkeypatch: pytest.MonkeyPatch):
    app, _ = make_app(monkeypatch, FakeLogfire(CONVERSATION + OTHER))
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        panel = app.query_one('#conversations', tui.OptionList)
        panel.highlighted = panel.get_option_index('p1')
        await pilot.pause(0.2)
        assert app.selected.id == 'e2'
        shown = [w.item.id for w in app.query(tui.Entry)]
        assert shown == ['p1', 'm1', 'm2', 'o1', 'p2', 'e2']
        assert str(app.query_one('#where', tui.Static).render()) == 'Fibonacci 30? ❯'


async def test_poll_reads_from_cursor_and_deduplicates(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION[:4])
    app, _ = make_app(monkeypatch, logfire)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert ('nodes', None) in logfire.calls  # the first load reads the whole window
        logfire.calls.clear()
        # A late record, older than the newest one seen but inside the overlap, and one newer record.
        logfire.nodes += [row(T2, '10:05:00', 'p2', 'o1', 'prompt', 'and 31?'), row(T1, '10:00:08', 'x', 'm2', 'error')]
        app.poll()
        await pilot.pause(0.3)
        since = next(s for kind, s in logfire.calls if kind == 'nodes')
        assert since == datetime.fromisoformat('2026-11-18T09:58:09+00:00')  # cursor 10:00:09 minus two minutes
        assert set(app.nodes) == {'p1', 'm1', 'm2', 'o1', 'p2', 'x'}
        app.poll()  # the same records again
        await pilot.pause(0.3)
        assert len(app.nodes) == 6


async def test_fork_from_any_point_and_retry(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION)
    logfire.executions = [
        {'trace_id': T1, 'start_timestamp': '2026-11-18T10:00:03Z', 'code': '1 + 1', 'result': '{"output": "2"}'}
    ]
    app, launched = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        nodes = app.nodes
        prompt = app.fork_point(nodes['p2'])
        assert (prompt.target.id, prompt.text) == ('o1', 'and 31?')  # edit the follow-up, from the result before it
        assert app.fork_point(nodes['p1']).target is None  # editing the first prompt starts a new conversation
        assert app.fork_point(nodes['m2']).target.id == 'm2'
        code_run = next(i for i in app.items if isinstance(i, tui.Execution))
        assert app.fork_point(code_run).target.id == 'm1'  # the step before the request that ran it

        app.pick(nodes['p2'], None)
        app.query_one('#composer', tui.Input).value = 'and 32?'
        await pilot.press('enter')
        await pilot.press('ctrl+g')  # retry the failure: the same follow-up again
        await pilot.pause(0.2)
        assert launched == [('o1', 'and 32?'), ('o1', 'and 31?')]


async def test_archive_hides_the_conversation(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION + OTHER)

    async def archive(self: Any, root: str, archived: bool = True) -> None:
        logfire.archive_events.append(
            {'start_timestamp': '2026-11-18T11:00:00Z', 'root': root, 'archived': str(archived).lower()}
        )

    monkeypatch.setattr(tui.TraceStore, 'archive', archive)
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.press('ctrl+y')
        await pilot.pause(0.3)
        roots = lambda: list(app.rows)
        assert roots() == ['q1']
        assert app.selected.id == 'q1'  # the next conversation takes its place
        app.poll()  # the archive event comes back from Logfire, and stays applied
        await pilot.pause(0.3)
        assert roots() == ['q1']
        app.toggle_archived()
        assert sorted(roots()) == ['p1', 'q1']
        app.select(app.nodes['e2'])
        await pilot.press('ctrl+y')  # unarchive
        await pilot.pause(0.3)
        app.toggle_archived()
        assert sorted(roots()) == ['p1', 'q1']


async def test_switch_branches_from_the_palette(monkeypatch: pytest.MonkeyPatch):
    fork = [
        row('d' * 32, '10:10:00', 'f1', 'm1', 'fork', 'm-b', model='m-b'),
        row('d' * 32, '10:10:04', 'k1', 'f1', 'question', 'Show the code?'),
    ]
    app, _ = make_app(monkeypatch, FakeLogfire(CONVERSATION + fork), conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app.selected.id == 'k1'  # the newest branch
        assert [t.id for t in app.branches()] == ['k1', 'e2']
        hits = [hit async for hit in tui.Prompts(app.screen).search('branch')]
        hits = [hit for hit in hits if str(hit.match_display).startswith('Branch')]
        titles = [str(hit.match_display) for hit in hits]
        assert titles == ['Branch: ⑂ m-b → ? Show the code?', 'Branch: › and 31? → ✗ MontyRefusedError: unavailable']
        hits[1].command()
        await pilot.pause(0.2)
        assert [w.item.id for w in app.query(tui.Entry)][-2:] == ['p2', 'e2']
        assert str(app.query_one('#cost', tui.Static).render()).startswith('branch 2/2')


async def test_sidebar_marks_conversations_updated_since_viewed(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION + OTHER)
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        panel = app.query_one('#conversations', tui.OptionList)

        def rows() -> dict[str, tuple[str, str]]:
            shown = {}
            for root in app.rows:
                left, right = (list(map(str, c.cells)) for c in panel.get_option(root).prompt.columns)
                shown[root] = (left[0] + right[0], left[1] + ' ' + right[1])
            return shown

        assert not any('●' in meta for _, meta in rows().values())  # what is there at startup counts as seen
        assert list(rows()) == ['p1', 'q1']  # most recently updated first
        assert rows()['p1'] == ('Fibonacci 30? ✗', f'$0 {tui.when("2026-11-18T10:05:01+00:00")}')
        assert rows()['q1'][0] == 'other conversation ◌'  # no output yet: mid-turn

        logfire.nodes.append(row('e' * 32, '10:20:00', 'q2', 'q1', 'output', 'done'))  # another terminal
        logfire.nodes.append(row(T2, '10:20:01', 'p3', 'e2', 'prompt', 'retry'))  # the conversation on screen
        app.poll()
        await pilot.pause(0.3)
        assert '●' in rows()['q1'][1] and '●' not in rows()['p1'][1]
        assert '1 updated' in str(app.query_one('#status', tui.Static).render())

        await pilot.click('#conversations', offset=(5, 2 * panel.get_option_index('q1') + 2))  # below the title
        await pilot.pause(0.2)
        assert app.selected.id == 'q2' and '●' not in rows()['q1'][1]
        assert rows()['q1'][0] == 'other conversation ✓'
        assert app.focused is app.query_one('#composer')


async def test_code_runs_panel_lists_every_branch(monkeypatch: pytest.MonkeyPatch):
    fork = [
        row('d' * 32, '10:10:00', 'f1', 'm1', 'fork', 'm-b', model='m-b'),
        row('d' * 32, '10:10:04', 'k1', 'f1', 'question', 'Show the code?'),
    ]
    logfire = FakeLogfire(CONVERSATION + fork + OTHER)
    logfire.executions = [
        {'trace_id': T1, 'start_timestamp': '2026-11-18T10:00:03Z', 'code': 'fib(30)', 'result': '{"output": "1"}'},
        {
            'trace_id': 'd' * 32,
            'start_timestamp': '2026-11-18T10:10:02Z',
            'code': '\n1 / 0',
            'result': '{"error": "x"}',
        },
    ]
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.5)
        app.selected = None
        app.select(app.nodes['e2'])  # the branch without the second run
        await pilot.press('ctrl+t')
        await pilot.pause(0.2)
        panel = app.query_one('#code-runs', tui.OptionList)
        assert panel.has_class('shown') and panel.has_focus
        labels = [str(panel.get_option_at_index(i).prompt) for i in range(panel.option_count)]
        assert [label[0] for label in labels] == ['✓', '✗'] and labels[1].endswith('1 / 0')

        await pilot.press('enter')  # the highlighted, newest run: opens its branch
        await pilot.pause(0.3)
        assert app.selected.id == 'k1'
        assert isinstance(app.focused, tui.CodeEntry) and app.focused.item.code == '\n1 / 0'


@pytest.mark.parametrize(
    ('ago', 'shown'),
    [(0, 'now'), (45, '45s'), (300, '5m'), (7140, '1h'), (2 * 86400, '2d'), (20 * 86400, '2w'), (80 * 86400, 'Aug 30')],
)
def test_when_is_human_age(ago: int, shown: str):
    now = datetime.fromisoformat('2026-11-18T12:00:00+00:00')
    assert tui.when((now - timedelta(seconds=ago)).isoformat(), now) == shown


@pytest.mark.parametrize('encode', [json.dumps, lambda value: value], ids=['text', 'parsed'])
def test_request_rows_accept_text_or_parsed_json(encode: Any):
    newest = {'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}
    output = [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'done'}]}]
    request = tui.Request.from_row(
        {
            'trace_id': T1,
            'start_timestamp': '2026-11-18T10:00:01Z',
            'instructions': json.dumps([{'type': 'text', 'content': 'Be brief.'}]),
            'input': encode(newest),
            'output': encode(output),
        }
    )
    assert (request.instructions, request.input, request.output) == ('Be brief.', newest['parts'], output[0]['parts'])


@pytest.mark.parametrize('encode', [json.dumps, lambda value: value], ids=['text', 'parsed'])
async def test_checkpoint_files_load_from_text_or_parsed_json(encode: Any):
    files = {'notes.txt': {'text': 'hi'}}
    checkpoint = row(T1, '10:00:01', 'ab' * 16, 'p1', 'question', 'Which date?', files=encode(files))
    state = await tui.TraceStore(FakeLogfire([checkpoint])).load('ab' * 16)
    assert state.files == {'notes.txt': 'hi'}


def test_span_url_filters_to_the_span_around_its_time():
    url = tui.span_url('https://logfire.example/org/proj', T1, '9fd776be67b6dac6', '2026-11-18T15:02:48+00:00')
    assert url.startswith(
        'https://logfire.example/org/proj?q=trace_id%3D%27' + T1 + '%27+and+span_id%3D%279fd776be67b6dac6%27'
    )
    assert '&traceId=' + T1 in url and '&spanId=9fd776be67b6dac6' in url
    assert 'since=2026-11-18T14%3A32%3A48%2B00%3A00' in url and 'until=2026-11-18T15%3A32%3A48%2B00%3A00' in url
    assert 'span_id' not in tui.span_url('https://logfire.example/org/proj', T1, None, '2026-11-18T15:02:48+00:00')


async def test_option_click_opens_the_span(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION)
    logfire.project_url = lambda: _async('https://logfire.example/org/proj')
    opened: list[str] = []
    monkeypatch.setattr(tui.webbrowser, 'open', opened.append)
    app, launched = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.pause(0.3)
        entry = next(w for w in app.query(tui.Entry) if w.item.id == 'o1')
        await pilot.click(entry, meta=True)
        await pilot.pause(0.2)
        assert opened == [tui.span_url('https://logfire.example/org/proj', T1, 'span-o1', entry.item.at)]
        assert app.fork is None and not launched  # not a pick


async def _async(value: Any) -> Any:
    return value


async def test_turn_from_a_picked_step_follows_the_new_branch(monkeypatch: pytest.MonkeyPatch):
    app, launched = make_app(monkeypatch, FakeLogfire(CONVERSATION), conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app.selected.id == 'e2'
        app.pick(app.nodes['m1'], None)
        await pilot.press('enter')  # re-run the model step
        await pilot.pause(0.2)
        assert launched == [('m1', '')]
        assert app.selected.id == 'm1' and app.fork is None
        assert [w.item.id for w in app.query(tui.Entry)] == ['p1', 'm1']

        await pilot.press('ctrl+n')
        app.query_one('#composer', tui.Input).value = 'fresh start'
        await pilot.press('enter')
        await pilot.pause(0.2)
        assert launched[-1] == (None, 'fresh start') and app.selected is None
