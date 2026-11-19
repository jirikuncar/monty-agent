"""The TUI against an in-memory stand-in for Logfire: no login, sandbox or network."""

import io
import json
import time
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

pytest.importorskip('textual')

from textual._xterm_parser import XTermParser

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
        self.variables: dict[tuple[str, str], Any] | None = {}
        """Managed variables by (name, label); None when the token lacks their scopes."""
        self.user: str | None = 'ada@example.com'

    async def __call__(self, sql: str, params: dict, since: datetime | None = None) -> list[dict]:
        kind = 'cost' if 'per_service' in sql else 'code' if 'child sandbox' in sql else 'other'
        kind = 'requests' if 'gen_ai' in sql else 'archive' if "archive' IS NOT NULL" in sql else kind
        kind = 'nodes' if kind == 'other' else kind
        self.calls.append((kind, since))
        if kind == 'cost':
            return [{'trace_id': t, 'cost': c} for t, c in self.costs.items() if 'IN (' not in sql or t in sql]
        rows = {'code': self.executions, 'requests': [], 'archive': self.archive_events, 'nodes': self.nodes}[kind]
        return [r for r in rows if since is None or datetime.fromisoformat(r['start_timestamp']) >= since]

    async def whoami(self) -> str | None:
        return self.user

    async def read_variable(self, name: str, label: str) -> Any:
        if self.variables is None:
            raise tui.NotGranted('log in again to grant project:read_variables')
        return self.variables.get((name, label))

    async def write_variable(self, name: str, label: str, value: Any, description: str) -> None:
        if self.variables is None:
            raise tui.NotGranted('log in again to grant project:write_variables')
        self.variables[name, label] = value

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
        assert app.query_one('#prompt').border_title == ' Fibonacci 30? '


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
        app.query_one('#composer', tui.Composer).value = 'and 32?'
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
        assert str(app.query_one('#cost', tui.Static).render()).startswith('branch 1/2')


def text(renderable: Any) -> str:
    console = Console(width=40, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


async def test_branches_panel_lists_the_open_conversations_branches(monkeypatch: pytest.MonkeyPatch):
    fork = [
        row('d' * 32, '10:10:00', 'f1', 'm1', 'fork', 'gateway/anthropic:m-b', model='m-b'),
        row('d' * 32, '10:10:04', 'k1', 'f1', 'question', 'Show the code?'),
    ]
    app, _ = make_app(monkeypatch, FakeLogfire(CONVERSATION + fork + OTHER), conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        conversations = app.query_one('#conversations', tui.OptionList)
        branches = app.query_one('#branches', tui.OptionList)
        ids = lambda panel: [panel.get_option_at_index(i).id for i in range(panel.option_count)]
        on = lambda: branches.get_option_at_index(branches.highlighted).id
        assert ids(conversations) == ['p1', 'q1']  # one row per conversation
        assert '$0 · ⑂2' in text(conversations.get_option_at_index(0).prompt)
        assert branches.has_class('shown') and ids(branches) == ['m2', 'f1'] and on() == 'f1'
        assert text(branches.get_option_at_index(1).prompt).startswith('└ ⑂ re-run m-b')

        app.action_focus_tree()
        await pilot.press('left')  # the sibling branch, at its newest node
        assert (app.selected.id, on()) == ('e2', 'm2')
        await pilot.press('left')
        assert app.selected.id == 'k1'
        branches.focus()
        await pilot.press('up')  # moving onto a branch row opens that branch
        await pilot.pause(0.2)
        assert (app.selected.id, on()) == ('e2', 'm2')

        app.select(app.nodes['q1'])  # a conversation without branches hides the panel
        assert not branches.has_class('shown') and ids(branches) == []


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

        # The times are terminal hyperlinks to the same spans: cmd-click opens them without the app.
        url = tui.span_url('https://logfire.example/org/proj', T1, 'span-o1', entry.item.at)
        assert url in entry.border_subtitle and '10:00:09' in entry.border_subtitle
        marker = next(w for w in app.query(tui.Entry) if w.item.id == 'm1')  # a one-line model step
        step = tui.span_url('https://logfire.example/org/proj', T1, 'span-m1', marker.item.at)
        assert step in marker.render().markup


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
        app.query_one('#composer', tui.Composer).value = 'fresh start'
        await pilot.press('enter')
        await pilot.pause(0.2)
        assert launched[-1] == (None, 'fresh start') and app.selected is None


async def test_prompt_title_takes_at_most_half_the_width(monkeypatch: pytest.MonkeyPatch):
    long = [row(T1, '10:00:00', 'p1', None, 'prompt', 'Show how to numerically solve ' * 6)]
    app, _ = make_app(monkeypatch, FakeLogfire(long), conversation='p1')
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause(0.3)
        title = app.query_one('#prompt').border_title
        assert title.startswith(' Show how') and title.endswith('… ') and len(title) <= 40


async def test_drag_resizes_the_sidebar_and_its_panels(monkeypatch: pytest.MonkeyPatch):
    app, launched = make_app(monkeypatch, FakeLogfire(CONVERSATION + OTHER), conversation='p1')
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        sidebar = app.query_one('#sidebar', tui.Sidebar)
        await pilot.mouse_down(sidebar, offset=(sidebar.region.width - 1, 5))  # the │ edge
        assert sidebar.has_class('resizing')
        await pilot.hover(sidebar, offset=(50, 5))
        await pilot.mouse_up(sidebar, offset=(50, 5))
        await pilot.pause()
        assert sidebar.region.width == 51 and not sidebar.has_class('resizing')
        await pilot.mouse_down(sidebar, offset=(sidebar.region.width - 1, 5))
        await pilot.hover(sidebar, offset=(100, 5))  # at most half the screen
        await pilot.mouse_up(sidebar, offset=(100, 5))
        await pilot.pause()
        assert sidebar.region.width == 60

        await pilot.press('ctrl+t')
        runs = app.query_one('#code-runs', tui.SidePanel)
        await pilot.pause()
        top = runs.region.y
        await pilot.mouse_down(runs, offset=(5, 0))  # its top line
        await pilot.hover(None, offset=(5, top - 2))  # screen coordinates: the panel moves while dragged
        await pilot.mouse_up(None, offset=(5, top - 2))
        await pilot.pause()
        assert runs.region.y == top - 2 and runs.styles.height.value == runs.region.height
        assert app.fork is None and not launched  # dragging picked nothing


async def test_composer_takes_several_lines(monkeypatch: pytest.MonkeyPatch):
    app, launched = make_app(monkeypatch, FakeLogfire(CONVERSATION), conversation='p1')
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        composer = app.query_one('#composer', tui.Composer)
        composer.focus()
        composer.value = ''
        await pilot.press(*'one', 'alt+enter', *'two', '\\', 'enter', *'three')
        assert [e.key for e in XTermParser().feed('\x1b\r')] == ['ctrl+j']  # option+Enter in iTerm2, tmux
        app.post_message(tui.events.Paste('\nfour\nfive'))  # as a terminal paste arrives
        await pilot.pause()
        assert composer.text == 'one\ntwo\nthree\nfour\nfive'
        await pilot.press('enter')
        await pilot.pause(0.2)
        assert launched[-1][1] == 'one\ntwo\nthree\nfour\nfive' and composer.text == ''


async def drag_sidebar(pilot: Any, app: tui.MontyTui, to: int) -> None:
    sidebar = app.query_one('#sidebar', tui.Sidebar)
    await pilot.mouse_down(sidebar, offset=(sidebar.region.width - 1, 5))
    await pilot.hover(None, offset=(to, 6))
    await pilot.mouse_up(None, offset=(to, 6))
    await pilot.pause(0.2)


async def test_layout_is_a_managed_variable(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION)
    ada, bob = (tui.SETTINGS, 'user:ada@example.com'), (tui.SETTINGS, 'user:bob@example.com')
    # Merged over the defaults: keys it lacks keep them, ones the TUI no longer has are dropped.
    logfire.variables = {ada: {'code-runs': 12, 'gone': 3}, bob: {'sidebar': 70}}
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert app.query_one('#sidebar').region.width == 40
        assert app.query_one('#code-runs').styles.height.value == 12
        await drag_sidebar(pilot, app, 29)
        assert logfire.variables[ada] == {'sidebar': 30, 'branches': None, 'code-runs': 12}
        assert logfire.variables[bob] == {'sidebar': 70}  # another user's label is untouched


async def test_layout_of_a_service_token_is_kept_for_the_session(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION)
    logfire.user = None
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        await drag_sidebar(pilot, app, 29)
        assert app.query_one('#sidebar').region.width == 30 and logfire.variables == {}


async def test_layout_without_the_variables_scopes(monkeypatch: pytest.MonkeyPatch):
    logfire = FakeLogfire(CONVERSATION)
    logfire.variables = None
    app, _ = make_app(monkeypatch, logfire, conversation='p1')
    lines: list[str] = []
    monkeypatch.setattr(app, 'log_line', lines.append)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        await drag_sidebar(pilot, app, 44)
        await drag_sidebar(pilot, app, 50)
        assert app.query_one('#sidebar').region.width == 51  # dragging still works, for this session
        assert [line for line in lines if 'layout' in line] == [
            'layout kept for this session only: log in again to grant project:read_variables'
        ]


class FakeMcp(tui.McpQuery):
    def __init__(self, tools: set[str], variables: dict[str, Any]):
        super().__init__('https://logfire.example', 'token')
        self.headers['mcp-session-id'] = 's'
        self.listed, self.variables, self.sent = tools, variables, []

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == 'tools/list':
            return {'tools': [{'name': name} for name in self.listed]}
        name, args = params['name'], params['arguments']
        self.sent.append((name, args.get('action')))
        if name == 'token_info':
            return {'content': [{'text': json.dumps({'user_email': 'ada@example.com', 'user_name': 'ada'})}]}
        if name == 'variable_get':
            if args['name'] not in self.variables:
                return {'content': [{'text': "Variable 'x' not found"}], 'isError': True}
            labels = {k: {'serialized_value': json.dumps(v)} for k, v in self.variables[args['name']].items()}
            return {'content': [{'text': json.dumps({'definition': {'labels': labels}})}]}
        if args['action'] == 'create':
            self.variables[args['name']] = {}
        elif args['name'] not in self.variables:
            return {'content': [{'text': "Variable 'x' not found"}], 'isError': True}
        else:
            for label in args['version']['labels']:
                self.variables[args['name']][label] = args['version']['value']
        return {'content': [{'text': '{}'}]}


async def test_mcp_variables():
    mcp = FakeMcp({'token_info', 'variable_get', 'variable_manage'}, {})
    assert await mcp.whoami() == 'ada@example.com'
    assert await mcp.read_variable('layout', 'user:ada') is None
    await mcp.write_variable('layout', 'user:ada', {'sidebar': 30}, 'the layout')
    await mcp.write_variable('layout', 'user:bob', {'sidebar': 50}, 'the layout')
    assert await mcp.read_variable('layout', 'user:ada') == {'sidebar': 30}
    assert await mcp.read_variable('layout', 'user:eve') is None
    actions = [action for _, action in mcp.sent[2:6]]
    assert actions == ['create_version', 'create', 'create_version', 'create_version']

    readonly = FakeMcp({'variable_get'}, {'layout': {'user:ada': {'sidebar': 30}}})
    assert await readonly.read_variable('layout', 'user:ada') == {'sidebar': 30}
    with pytest.raises(tui.NotGranted, match='project:write_variables'):
        await readonly.write_variable('layout', 'user:ada', {}, 'the layout')
    with pytest.raises(tui.NotGranted, match='project:read_variables'):
        await FakeMcp(set(), {}).read_variable('layout', 'user:ada')


async def test_entries_name_who_sent_them(monkeypatch: pytest.MonkeyPatch):
    nodes = [
        row(T1, '10:00:00', 'p1', None, 'prompt', 'Fibonacci 30?', user='ada@example.com'),
        row(T1, '10:00:05', 'q1', 'p1', 'question', 'Which index?'),
        row(T1, '10:00:09', 'a1', 'q1', 'answer', '30', user='bob@example.com'),
        row(T1, '10:00:12', 'f1', 'a1', 'fork', 'm-b', user='bob@example.com'),
        row(T1, '10:00:15', 'p2', 'f1', 'prompt', 'and 31?'),  # recorded before senders were
    ]
    app, _ = make_app(monkeypatch, FakeLogfire(nodes), conversation='p1')
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert app.user == 'ada@example.com'
        titles = {w.item.id: w.border_title or str(w.render()) for w in app.query(tui.Entry)}
        assert titles['p1'].endswith(' You') and titles['p2'].endswith(' Prompt')
        assert titles['a1'].endswith(' bob@example.com answered')
        assert 'Re-run with m-b · bob@example.com' in titles['f1']
