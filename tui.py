"""Browse, answer, fork and continue agent conversations kept in Logfire; nothing is stored locally.

    uv run --extra tui tui.py --remote https://logfire-us.pydantic.dev/ --model gateway/anthropic:claude-haiku-5-5

Logs in with OAuth (device flow), runs every turn in Agent Sandboxes, saves each checkpoint as a Logfire record and
reads the tree back with SQL through the Logfire MCP server. The token is refreshed before it expires; if that fails,
log in again from the command palette (ctrl+p, "Log in again"). With `--reload`, saving a module next to this one
reloads the code and restarts the app with the same login.

The sidebar lists conversations, the open one unfolded into its branches; the main pane shows the selected branch,
live while a turn runs. Pick a node, then type in the composer:

- nothing selected (ctrl+n), or a result: a prompt (a new conversation, or the next turn of that one)
- a question: the answer (answering again branches)
- a model step: Enter to re-run from there, optionally with another model name

In the command palette (ctrl+p), text that matches no command runs as a new prompt, or continues the selection.
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import webbrowser
from collections.abc import Callable, Iterable, Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlencode
from uuid import uuid4

import logfire
import requests
from pydantic_core import to_jsonable_python
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Markdown, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from host import Sandbox, Suspended, configure_logfire, conversation, sandbox_key
from oauth import Login, login, refresh
from trace_store import Execution, McpQuery, Node, Request, TraceStore, iso, parsed

HERE = Path(__file__).parent
RELOAD = 'reload'
MODULES = ('oauth', 'host', 'trace_store', 'tui')
"""Reloaded in dependency order."""
OVERLAP = timedelta(minutes=2)
"""How far before the cursor a poll reads again: records can become queryable out of order (export batches,
ingestion lag), so a poll that started exactly at the cursor could skip one."""
REFRESH_BEFORE = 300
"""Seconds before expiry to refresh, so a turn started just before still has a valid token."""


def export_with(token: Callable[[], str]) -> None:
    """Make the Logfire SDK send each export with the current token instead of the one it was configured with.

    The SDK fixes the token in every exporter's request headers at `configure`, and configuring again would shut
    down the tracer provider under open spans. A `requests` auth hook runs after the headers are merged, so it wins.
    Call before `configure`.
    """
    from logfire._internal import config

    def auth(request: requests.PreparedRequest) -> requests.PreparedRequest:
        request.headers['Authorization'] = token()
        return request

    class Session(config.OTLPExporterHttpSession):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.auth = auth

    config.OTLPExporterHttpSession = Session


ICONS = {
    'prompt': ('›', 'bold cyan'),
    'answer': ('›', 'bold cyan'),
    'fork': ('⑂', 'magenta'),
    'model': ('·', 'dim'),
    'question': ('?', 'bold yellow'),
    'output': ('✓', 'bold green'),
    'error': ('✗', 'bold red'),
}
SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
STANDING = ('output', 'question', 'error')  # a conversation's newest node when no turn is running
NEXT = {
    None: 'ask anything',
    'output': 'reply',
    'question': 'answer (answering again branches)',
    'model': 'enter re-runs this step; a model name forks with it',
}


def span_url(project_url: str, trace: str, span: str | None, at: str) -> str:
    """Logfire's live view filtered to the span (or its trace), around the time it started."""
    moment = datetime.fromisoformat(at)
    query = f"trace_id='{trace}'" + (f" and span_id='{span}'" if span else '')
    params = {'q': query, 'traceId': trace, **({'spanId': span} if span else {})}
    params |= {
        'since': (moment - timedelta(minutes=30)).isoformat(),
        'until': (moment + timedelta(minutes=30)).isoformat(),
    }
    return f'{project_url}?{urlencode(params)}'


def when(at: str, now: datetime | None = None) -> str:
    """How long ago: now, 45s, 5m, 3h, 2d, 6w, then the date."""
    moment = datetime.fromisoformat(at)
    seconds = ((now or datetime.now(UTC)) - moment).total_seconds()
    for unit, size in (('w', 604800), ('d', 86400), ('h', 3600), ('m', 60), ('s', 1)):
        if seconds >= size:
            return f'{int(seconds // size)}{unit}' if unit != 'w' or seconds < 9 * size else f'{moment:%b %d}'
    return 'now'


def one_line(text: str, width: int = 60) -> str:
    text = ' '.join(text.replace('**', '').replace('`', '').split())
    return text if len(text) <= width else text[: width - 1] + '…'


def label(node: Node) -> Text:
    icon, style = ICONS[node.kind]
    text = f'step · {node.attrs.get("messages", "?")} messages' if node.kind == 'model' else one_line(node.text)
    return Text.assemble((icon, style), ' ', (text, 'dim' if node.kind == 'model' else ''))


Item = Node | Execution
"""Anything in the transcript; each one can be forked from (`MontyTui.fork_point`)."""


class Pickable(Static):
    """A transcript entry: click it, or focus it and press Enter, to fork from there; option-click (or ctrl-click)
    opens its span in Logfire."""

    can_focus = True
    BINDINGS: ClassVar = [Binding('enter', 'pick', 'Fork here')]
    item: Item

    def on_click(self, event: events.Click) -> None:
        app = self.app
        if (event.meta or event.ctrl) and isinstance(app, MontyTui):
            app.open_in_logfire(getattr(self, 'request', None) or self.item)
        else:
            self.action_pick()

    def action_pick(self) -> None:
        app = self.app
        if isinstance(app, MontyTui):
            app.pick(self.item, self)


class Entry(Pickable):
    """One node in the transcript; a model step shows the request it was taken before, expanded with ctrl+o."""

    def __init__(self, node: Node, request: Request | None = None, *, sub_agent: bool = False, expanded: bool = False):
        icon, _ = ICONS[node.kind]
        self.request, self.sub_agent = request, sub_agent
        title = {
            'prompt': 'You',
            'answer': 'You answered',
            'fork': f'Re-run with {node.text}',
            'model': f'Model step · {node.attrs.get("messages", "?")} messages',
            'question': 'Agent asks',
            'output': 'Result',
            'error': 'Failed',
        }[node.kind]
        self.item = node
        if node.kind == 'model' and request is not None:
            who = f'{"↳ " if sub_agent else ""}{agent_name(request.instructions)}'
            title = f'Model step · {who} · {node.attrs.get("messages", "?")} messages'
            if expanded:
                super().__init__(classes='entry model expanded sub' if sub_agent else 'entry model expanded')
                self.border_title = f'{icon} {title}'
                self.border_subtitle = node.at[11:19]
                self.body = ''
                return
        if node.kind in ('fork', 'model'):
            # A one-line marker between the bordered messages.
            super().__init__(f'{icon} {title} · {node.at[11:19]}', classes=f'entry {node.kind}', markup=False)
            self.body = ''
            return
        super().__init__(classes=f'entry {node.kind}')
        self.border_title = f'{icon} {title}'
        self.border_subtitle = node.at[11:19]
        self.body = node.text

    def compose(self) -> ComposeResult:
        if self.has_class('expanded') and self.request is not None:
            yield from request_parts(self.request, sub_agent=self.sub_agent)
        elif self.body and self.has_class('output'):
            yield Markdown(self.body)
        elif self.body:
            yield Static(self.body, markup=False)


class CodeEntry(Pickable):
    """Code the agent ran in a child sandbox, and its result."""

    def __init__(self, execution: Execution):
        super().__init__(classes='entry code')
        self.item = execution
        self.border_title = '⌘ Ran code'
        self.border_subtitle = execution.at[11:19]

    def compose(self) -> ComposeResult:
        yield Static(python(self.item.code), classes='source')
        try:
            result = parsed(self.item.result, {})
        except ValueError:
            result = {'output': self.item.result}
        if not isinstance(result, dict):
            result = {'output': result}
        lines = [f'{k}: {str(v).rstrip()}' for k, v in result.items() if v not in ('', None)]
        yield Static(
            '\n'.join(lines) or '(no output)', markup=False, classes='error' if 'error' in result else 'result'
        )


def python(code: str) -> Syntax:
    return Syntax(code, 'python', theme='ansi_dark', line_numbers=True, background_color='default')


def agent_name(instructions: str) -> str:
    return one_line(instructions.split('. ')[0], 48) or 'agent'


def request_parts(request: Request, *, sub_agent: bool) -> Iterator[Widget]:
    """What a model request added to the conversation: tool results (or a sub-agent's task) in, the reply out."""
    for part in request.input:
        if part.get('type') == 'tool_call_response':
            result = part.get('result')
            if isinstance(result, list):
                result = ' '.join(r.get('text', '') for r in result if isinstance(r, dict))
            text = result if isinstance(result, str) else json.dumps(result)
            yield Static(f'← {part.get("name")}: {clip(text)}', markup=False, classes='returned')
        elif part.get('type') == 'text' and sub_agent:
            yield Static(f'task: {clip(part.get("content", ""))}', markup=False, classes='returned')
    for part in request.output:
        kind = part.get('type')
        if kind == 'text' and part.get('content'):
            yield Markdown(part['content'])
        elif kind == 'thinking' and part.get('content'):
            yield Static(clip(part['content']), markup=False, classes='thinking')
        elif kind == 'tool_call':
            arguments = part.get('arguments') or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {'arguments': arguments}
            name = part.get('name', '?')
            if name == 'run_code' and isinstance(arguments, dict) and 'code' in arguments:
                files = arguments.get('files')
                yield Static(f'→ run_code{f" with {files}" if files else ""}', markup=False, classes='call')
                yield Static(python(arguments['code']))
            else:
                yield Static(f'→ {name}({clip(json.dumps(arguments), 300)})', markup=False, classes='call')


def clip(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[: limit - 1] + '…'


def price(cost: float | None) -> str:
    return '' if not cost else f'${cost:.4f}' if cost < 1 else f'${cost:.2f}'


def on_path[T: (Execution, Request)](path: list[Node], executions: list[T]) -> list[T]:
    """Items of the path's turns up to the path's last node in each turn; later ones belong to other branches."""
    last = {}
    for node in path:
        if node.trace:
            last[node.trace] = max(last.get(node.trace, ''), node.at)
    return [e for e in executions if e.trace in last and e.at <= last[e.trace]]


@dataclass
class Run:
    title: str
    started: float = field(default_factory=time.monotonic)


def describe(item: Item) -> str:
    if isinstance(item, Node):
        return label(item).plain
    return f'code run: {one_line(item.code, 50)}'


@dataclass
class Branch:
    """Nodes from `start` to the next branch point; `start` is the root or one of several children."""

    start: Node
    nodes: list[Node]
    """Everything below `start`, in all its branches."""
    depth: int
    guides: str
    """Tree lines before the row, from the branches it hangs under."""
    leaf: bool
    siblings: list[Node]

    @property
    def tip(self) -> Node:
        return max(self.nodes, key=lambda n: n.at)


@dataclass
class ForkPoint:
    target: Node | None
    """The checkpoint the new branch starts from; None starts a new conversation."""
    verb: str
    text: str
    """Prefilled into the composer, e.g. the prompt being edited."""


class Prompts(Provider):
    """Palette fallback: whatever matches no command can be run as a prompt."""

    async def search(self, query: str) -> Hits:
        app = self.app
        text = query.strip()
        if not isinstance(app, MontyTui) or not text:
            return
        yield Hit(0.001, Content.assemble(('New conversation: ', 'bold'), text), partial(app.launch, None, text), text)
        if app.selected is not None and (point := app.fork_point(app.selected)) is not None:
            verb = Content.assemble((f'{point.verb}: ', 'bold'), text)
            yield Hit(0, verb, partial(app.launch, point.target, text), text)
        matcher = self.matcher(query)
        for tip in app.branches():
            title = f'Branch: {app.branch_title(tip)}'
            if score := matcher.match(title):
                yield Hit(score, matcher.highlight(title), partial(app.show_branch, tip), help=tip.at[11:19])
        for item in app.items:
            if app.fork_point(item) is not None:
                title = f'Fork from {describe(item)}'
                if score := matcher.match(title):
                    yield Hit(score, matcher.highlight(title), partial(app.pick, item, None), help=item.at[11:19])

    async def discover(self) -> Hits:
        app = self.app
        if isinstance(app, MontyTui):
            yield DiscoveryHit('New conversation', app.action_new, help='Type a prompt here to start it right away')


class MontyTui(App[str | None]):
    TITLE = 'monty-agent'
    COMMANDS = App.COMMANDS | {Prompts}
    CSS = """
    #sidebar { width: 40; border-right: solid $panel-lighten-1; }
    #conversations { height: 1fr; background: $background; border: none; border-top: solid $panel-lighten-1; }
    #conversations > .option-list--option { padding: 0 1 0 0; }
    #transcript { padding: 0 1; }
    .empty { color: $text-muted; padding: 1 2; }
    .entry { border: round $panel; padding: 0 1; margin: 0 0 1 0; height: auto; }
    .entry.prompt, .entry.answer { border: round $accent; }
    .entry.question { border: round $warning; }
    .entry.output { border: round $success; }
    .entry.error { border: round $error; color: $error; }
    .entry.model, .entry.fork { border: none; padding: 0 1; height: 1; color: $text-muted; }
    .entry Markdown { margin: 0; padding: 0; }
    .entry:focus { background: $boost; }
    .entry.picked { border: double $accent; }
    .entry.model.picked, .entry.fork.picked { height: auto; color: $text; }
    .entry.model.expanded { height: auto; border: round $panel-lighten-2; color: $text; padding: 0 1; }
    .entry.model.sub { margin-left: 4; }
    .entry .returned { color: $text-muted; }
    .entry .call { color: $secondary; }
    .entry .thinking { color: $text-muted; text-style: italic; }
    .entry.code { border: round $primary; }
    .entry.code .result { color: $text-muted; border-top: dashed $panel; }
    .entry.code .error { color: $error; border-top: dashed $panel; }
    #cost { height: 1; padding: 0 1; color: $text-muted; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #prompt { height: 3; border-top: solid $panel-lighten-1; border-bottom: solid $panel-lighten-1; }
    #where { width: auto; max-width: 36; padding: 0 1; color: $accent; text-style: bold; }
    #composer { border: none; height: 1; padding: 0; background: $background; }
    #composer:focus { border: none; }
    #code-runs { height: 1fr; border-top: solid $panel-lighten-1; background: $background; display: none; }
    #code-runs.shown { display: block; }
    #log { height: 8; border-top: solid $panel-lighten-1; display: none; }
    #log.shown { display: block; }
    """
    BINDINGS: ClassVar = [
        Binding('ctrl+n', 'new', 'New'),
        Binding('ctrl+r', 'refresh', 'Refresh'),
        Binding('ctrl+l', 'toggle_log', 'Activity'),
        Binding('ctrl+o', 'toggle_expand', 'Expand'),
        Binding('ctrl+t', 'toggle_code', 'Code runs'),
        Binding('ctrl+g', 'retry', 'Retry failed'),
        Binding('ctrl+y', 'toggle_archive', 'Archive'),  # ctrl+b is the tmux prefix
        Binding('left', 'sibling(-1)', 'Branch', show=False),
        Binding('right', 'sibling(1)', 'Branch', show=False),
        Binding('escape', 'focus_tree', 'Conversations', show=False),
        Binding('ctrl+q', 'quit', 'Quit'),
    ]

    def __init__(self, args: argparse.Namespace, session: Login):
        super().__init__()
        # Defaults for options added since `args` was parsed: a reload keeps the process that parsed them.
        self.args = argparse.Namespace(**(vars(parse_args([])) | vars(args)))
        self.login = session
        self.sql = McpQuery(session.backend, session.access_token)
        self.nodes: dict[str, Node] = {}
        self.costs: dict[str, float] = {}
        self.executions: dict[str, list[Execution]] = {}
        self.cursor: str | None = None
        """`at` of the newest record a query returned; polls read from it."""
        self.requests: dict[str, list[Request]] = {}
        self.expanded = False
        self.items: list[Item] = []
        """The transcript on screen, oldest first."""
        self.step_requests: dict[str, tuple[Request, bool]] = {}
        self.last_step: str | None = None
        self.latest: dict[str, str] = {}
        """Conversation (root node) -> `at` of its newest node."""
        self.seen: dict[str, str] | None = None
        """Conversation -> newest `at` shown in this terminal; set to everything there at the first load."""
        self.rows: dict[str, int] = {}
        self.order: list[str] = []
        self.project_url: str | None = None
        self.runs_shown: list[Execution] = []
        self.archived: dict[str, tuple[str, bool]] = {}
        """Root node -> (when, archived), from the newest archive event."""
        self.show_archived = False
        self.fork: ForkPoint | None = None
        """Where Enter in the composer branches from, when picked in the transcript; else the selection decides."""
        self.selected: Node | None = None
        self.runs: dict[int, Run] = {}
        self._tick = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id='sidebar'):
                conversations = OptionList(id='conversations')
                conversations.border_title = 'Conversations'
                yield conversations
                yield OptionList(id='code-runs')
            with Vertical():
                yield Static(id='cost')
                yield VerticalScroll(id='transcript')
        yield RichLog(id='log', wrap=True, markup=False)
        yield Static(id='status')
        with Horizontal(id='prompt'):
            yield Static('❯', id='where')
            yield Input(id='composer')
        yield Footer()

    async def on_mount(self) -> None:
        self.use(self.login)
        self.keep_fresh()
        if self.args.reload:
            self.watch_code()
        self.set_interval(0.1, self.show_status)
        self.set_interval(30, self.relabel)  # keeps the 5m, 1h ages current
        if self.args.poll:
            self.set_interval(self.args.poll, self.poll)
        self.select(None)
        self.action_refresh()

    async def on_unmount(self) -> None:
        await self.sql.aclose()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand('New conversation', 'Clear the selection and type a prompt', self.action_new)
        yield SystemCommand('Refresh conversations', 'Query Logfire for every node again', self.action_refresh)
        yield SystemCommand('Toggle activity log', 'Show or hide the event log', self.action_toggle_log)
        focused = screen.focused if isinstance(screen.focused, Pickable) else None
        if focused is not None or self.selected is not None:
            target = (getattr(focused, 'request', None) or focused.item) if focused else self.selected
            yield SystemCommand(
                'Open in Logfire',
                'The focused entry (or the selection) in the browser; or option-click it',
                partial(self.open_in_logfire, target),
            )
        yield SystemCommand(
            'Toggle code runs', 'List the code run in every branch of this conversation', self.action_toggle_code
        )
        yield SystemCommand(
            'Expand model steps', 'Show each request: tool results in, reply out', self.action_toggle_expand
        )
        if (root := self.conversation_id()) is not None:
            if self.is_archived(root):
                yield SystemCommand('Unarchive conversation', 'Show it in the list again', self.unarchive)
            else:
                yield SystemCommand(
                    'Archive conversation', 'Hide it from the list (records stay); ctrl+y', self.archive
                )
        what = 'Hide' if self.show_archived else 'Show'
        yield SystemCommand(f'{what} archived conversations', 'Toggle archived ones in the list', self.toggle_archived)
        if (failed := self.last_failure()) is not None:
            yield SystemCommand('Retry last failed request', one_line(failed.text, 80), self.action_retry)
        if self.selected is not None:
            yield SystemCommand(
                'Copy conversation ID', 'For --conversation in another window', self.copy_conversation_id
            )
        if self.runs:
            yield SystemCommand('Cancel running turns', f'Stop {len(self.runs)} running turn(s)', self.cancel_runs)
        yield SystemCommand('Log in again', 'New device login, e.g. after a failed refresh', self.relogin)
        yield SystemCommand('Refresh token', 'Renew the access token now', self.refresh_now)

    def use(self, session: Login) -> None:
        """Switch every client to `session`: sandbox pools and models read the environment when a turn starts."""
        self.login = session
        os.environ.update(session.env())
        self.sql.set_token(session.access_token)
        self.sub_title = f'{self.args.model} · {self.args.script.name} · token until {datetime.fromtimestamp(session.expires_at):%H:%M}'

    @work(exclusive=True, group='token')
    async def keep_fresh(self) -> None:
        while True:
            await asyncio.sleep(max(self.login.expires_at - REFRESH_BEFORE - time.time(), 0))
            try:
                self.use(await refresh(self.login))
            except Exception as error:  # noqa: BLE001 - the palette offers a new login
                self.sub_title = 'token expired' if time.time() > self.login.expires_at else 'token refresh failed'
                self.log_line(f'{error}; log in again from the command palette (ctrl+p)')
                self.notify('Token refresh failed: ctrl+p, Log in again', severity='error', timeout=30)
                return
            self.log_line('token refreshed')

    @work(group='reload')
    async def watch_code(self) -> None:
        from watchfiles import PythonFilter, awatch

        async for _ in awatch(HERE, watch_filter=PythonFilter(), recursive=False):
            self.exit(RELOAD)
            return

    def refresh_now(self) -> None:
        self.login = replace(self.login, expires_at=time.time() + REFRESH_BEFORE)
        self.keep_fresh()

    @work(exclusive=True, group='token')
    async def relogin(self) -> None:
        def show(url: str, code: str) -> None:
            self.log_line(f'approve in your browser: {url} (code {code})')
            self.notify(f'Approve the login in your browser, code {code}', timeout=60)

        try:
            session = await login(self.args.remote, show, gateway=self.args.gateway)
        except Exception as error:  # noqa: BLE001 - shown in the log pane, the old token stays in use
            self.log_line(f'login failed: {error}')
            return
        self.use(session)
        self.log_line('logged in')
        self.keep_fresh()
        self.action_refresh()

    def log_line(self, text: str) -> None:
        self.query_one('#log', RichLog).write(f'{datetime.now():%H:%M:%S} {text}')

    def show_status(self) -> None:
        self._tick += 1
        if not self.runs:
            updated = f'{n} updated · ' if (n := self.unread()) else ''
            text = f'idle · {updated}{len(self.latest)} conversations · ctrl+p commands, or type anything to run it'
        else:
            now = time.monotonic()
            runs = ', '.join(f'{one_line(r.title, 30)} {now - r.started:.0f}s' for r in self.runs.values())
            text = f'{SPINNER[self._tick % len(SPINNER)]} {len(self.runs)} running: {runs}'
        for status in self.query('#status').results(Static):  # none while the app shuts down
            status.update(text)

    def action_new(self) -> None:
        self.query_one('#conversations', OptionList).highlighted = None
        self.select(None)
        self.query_one('#composer', Input).focus()

    def action_toggle_log(self) -> None:
        self.query_one('#log').toggle_class('shown')

    def action_toggle_code(self) -> None:
        runs = self.query_one('#code-runs', OptionList)
        runs.toggle_class('shown')
        if runs.has_class('shown'):
            runs.focus()

    def conversation_runs(self) -> list[Execution]:
        """Every code run of the selected conversation, in all its branches, oldest first."""
        root = self.conversation_id()
        traces = {n.trace for n in self.subtree(self.nodes[root]) if n.trace} if root else set()
        return sorted((e for t in traces for e in self.executions.get(t, [])), key=lambda e: e.at)

    def show_runs(self, runs: list[Execution]) -> None:
        panel = self.query_one('#code-runs', OptionList)
        on_screen = {id(e) for e in self.items if isinstance(e, Execution)}
        options = []
        for index, run in enumerate(runs):
            try:
                failed = 'error' in parsed(run.result, {})
            except ValueError:
                failed = False
            first = next((line.strip() for line in run.code.splitlines() if line.strip()), '')
            prompt = Text.assemble(
                ('✗ ' if failed else '✓ ', 'red' if failed else 'green'),
                (f'{when(run.at)} ', 'dim'),
                (one_line(first, 26), '' if id(run) in on_screen else 'dim'),
            )
            options.append(Option(prompt, id=str(index)))
        highlighted = panel.highlighted
        panel.clear_options()
        panel.add_options(options)
        panel.border_title = f'Code runs · {len(runs)}'
        if options:
            panel.highlighted = min(highlighted, len(options) - 1) if highlighted is not None else len(options) - 1
        self.runs_shown = runs

    @on(OptionList.OptionSelected, '#code-runs')
    def on_run_selected(self, event: OptionList.OptionSelected) -> None:
        run = self.runs_shown[int(event.option.id or 0)]
        if not any(e is run for e in self.items):
            # Its branch: the newest node of the turn that ran it.
            turn = [n for n in self.nodes.values() if n.trace == run.trace]
            if turn:
                self.select(max(turn, key=lambda n: n.at))
        self.call_after_refresh(self.reveal_run, run)

    def reveal_run(self, run: Execution) -> None:
        for widget in self.query(CodeEntry):
            if widget.item is run:
                widget.scroll_visible(animate=False, top=True)
                widget.focus()

    def action_focus_tree(self) -> None:
        if self.fork is not None:
            self.unpick()
            return
        panel = self.query_one('#conversations', OptionList)
        if panel.highlighted is None and panel.option_count:
            panel.highlighted = 0  # so enter opens one
        panel.focus()

    def is_archived(self, root: str) -> bool:
        return self.archived.get(root, ('', False))[1]

    def archive(self) -> None:
        self.set_archived(True)

    def action_toggle_archive(self) -> None:
        if (root := self.conversation_id()) is not None:
            self.set_archived(not self.is_archived(root))

    def unarchive(self) -> None:
        self.set_archived(False)

    @work(group='archive')
    async def set_archived(self, archived: bool) -> None:
        root = self.conversation_id()
        if root is None:
            return
        await TraceStore(self.sql).archive(root, archived)
        self.archived[root] = (iso(datetime.now(UTC)), archived)  # before Logfire can be queried for it
        if archived and not self.show_archived:
            # Triage: the next conversation in the list takes its place.
            rows = list(self.rows)
            at = rows.index(root) if root in rows else 0
            rest = [r for r in rows[at + 1 :] + rows[:at][::-1] if r in self.nodes and not self.is_archived(r)]
            self.selected = None
            self.show_tree()
            if rest:
                self.select(max(self.subtree(self.nodes[rest[0]]), key=lambda n: n.at))
                self.show_tree()
            else:
                self.select(None)
        else:
            self.show_tree()
        title = one_line(self.nodes[root].text, 30)
        self.notify(f'Archived {title}; ctrl+p Show archived' if archived else f'Unarchived {title}')

    def toggle_archived(self) -> None:
        self.show_archived = not self.show_archived
        self.show_tree()

    def last_failure(self) -> Node | None:
        """The newest error in the selected conversation, or anywhere when none is selected."""
        root = self.conversation_id()
        nodes = self.subtree(self.nodes[root]) if root else list(self.nodes.values())
        return max((n for n in nodes if n.kind == 'error'), key=lambda n: n.at, default=None)

    def action_retry(self) -> None:
        """Run the failed request again: the same input, from the checkpoint it started at."""
        failed = self.last_failure()
        parent = self.nodes.get(failed.parent) if failed and failed.parent else None
        point = self.fork_point(parent) if parent else None
        if point is None:
            self.notify('No failed request to retry here', severity='warning')
            return
        text = point.text
        if point.target is not None and point.target.kind == 'model' and not text and failed is not None:
            # Same model as the failed turn: its prompt, answer or fork node recorded it.
            turn = [n for n in self.nodes.values() if n.trace == failed.trace and 'model' in n.attrs]
            text = turn[0].attrs['model'] if turn else ''
        if point.target is not None:
            self.show_tree(select=point.target.id)  # the transcript follows the retry
        self.log_line(f'retry: {one_line(failed.text if failed else "", 80)}')
        self.launch(point.target, text)

    def action_toggle_expand(self) -> None:
        self.expanded = not self.expanded
        self.select(self.selected)

    def fork_point(self, item: Item) -> ForkPoint | None:
        """Where a branch from `item` starts: the checkpoint just before it, and how the composer reads."""
        if isinstance(item, Node):
            parent = self.nodes.get(item.parent) if item.parent else None
            match item.kind:
                case 'prompt':
                    verb = 'Edit the prompt and run it again' if parent else 'Edit the prompt (a new conversation)'
                    return ForkPoint(parent, verb, item.text)
                case 'answer':
                    return ForkPoint(parent, 'Change the answer', item.text)
                case 'question':
                    return ForkPoint(item, 'Answer', '')
                case 'model':
                    return ForkPoint(item, 'Re-run from this step (Enter, or a model name)', '')
                case 'fork':
                    return ForkPoint(parent, 'Re-run this step (Enter, or a model name)', item.text)
                case 'output':
                    return ForkPoint(item, 'Continue from here', '')
                case _:
                    return self.fork_point(parent) if parent else None
        # A code run: re-run from the checkpoint taken before the request that asked for it.
        steps = [n for n in self.items if isinstance(n, Node) and n.kind == 'model' and n.trace == item.trace]
        steps = [n for n in steps if n.at <= item.at]
        return ForkPoint(steps[-1], 'Re-run from the model step before this', '') if steps else None

    def pick(self, item: Item, widget: Pickable | None) -> None:
        point = self.fork_point(item)
        if point is None:
            self.notify('No checkpoint before this to fork from', severity='warning')
            return
        self.unpick()
        self.fork = point
        if widget is None:  # from the palette
            widget = next((w for w in self.query(Pickable) if w.item is item), None)
        if widget is not None:
            widget.add_class('picked')
            widget.scroll_visible()
        composer = self.query_one('#composer', Input)
        composer.placeholder = f'⑂ {point.verb} · esc cancels'
        composer.value = point.text
        composer.focus()

    @work(group='links')
    async def open_in_logfire(self, item: Item | Request) -> None:
        """The item's span in the Logfire UI: the model request of a model step, the record of other nodes."""
        if item.trace is None:
            self.notify('Not in Logfire yet', severity='warning')
            return
        if self.project_url is None:
            try:
                self.project_url = await self.sql.project_url()
            except Exception as error:  # noqa: BLE001 - shown as a toast
                self.notify(f'No Logfire link: {error}', severity='error')
                return
        url = span_url(self.project_url, item.trace, item.span, item.at)
        self.log_line(f'open {url}')
        webbrowser.open(url)

    def unpick(self) -> None:
        self.fork = None
        for widget in self.query('.picked'):
            widget.remove_class('picked')
        self.show_where()

    def show_where(self) -> None:
        """The composer names the conversation it continues: its first prompt, shortened."""
        self.query_one('#composer', Input).placeholder = self.placeholder()
        root = self.conversation_id()
        title = one_line(self.nodes[root].text, 30) if root else 'new conversation'
        self.query_one('#where', Static).update(Text(f'{title} ❯'))

    def placeholder(self) -> str:
        if self.selected is None:
            return NEXT[None]
        point = self.fork_point(self.selected)
        return NEXT.get(self.selected.kind) or (f'⑂ {point.verb}' if point else 'Nothing to continue from here')

    def cancel_runs(self) -> None:
        self.workers.cancel_group(self, 'turns')
        self.runs.clear()
        self.log_line('cancelled running turns')

    def action_refresh(self) -> None:
        self.load(full=True)

    def poll(self) -> None:
        self.load(full=False)

    @work(exclusive=True, group='refresh')
    async def load(self, full: bool) -> None:
        """Merge nodes and costs from Logfire.

        A full load reads the whole window. A poll reads from the cursor (the newest record seen) minus `OVERLAP`,
        since a record can become queryable after newer ones; nodes are deduplicated by ID, and only the costs of
        the traces it saw are read again, from their first known node on.
        """
        since = None if full or self.cursor is None else datetime.fromisoformat(self.cursor) - OVERLAP
        store = TraceStore(self.sql)
        try:
            nodes, archived = await asyncio.gather(store.nodes(since=since), store.archived(since=since))
            if since is None:
                costs = await store.costs()
            else:
                traces = {n.trace for n in nodes if n.trace}
                known = [n.at for n in [*self.nodes.values(), *nodes] if n.trace in traces]
                costs = await store.costs(traces, since=datetime.fromisoformat(min(known))) if known else {}
        except Exception as error:  # noqa: BLE001 - shown in the log pane, the app keeps running
            self.log_line(f'query failed: {error}')
            self.notify(f'Query failed: {one_line(str(error))}', severity='error')
            return
        self.cursor = max([n.at for n in nodes] + ([self.cursor] if self.cursor else []), default=None)
        archive_changed = False
        for root, (at, value) in archived.items():
            if root not in self.archived or self.archived[root][0] < at:
                archive_changed |= self.archived.get(root, (at, False))[1] != value
                self.archived[root] = (at, value)
        first = not self.nodes
        new = sorted((n for n in nodes if n.id not in self.nodes), key=lambda n: n.at)
        for node in nodes:  # a node this terminal recorded learns its span ID from Logfire
            if (known := self.nodes.get(node.id)) is not None and known.span is None:
                known.span = node.span
        changed = {t for t in costs if costs[t] != self.costs.get(t)} | {n.trace for n in new if n.trace}
        if not new and not changed and not first and not archive_changed:
            return  # nothing to redraw: keep the transcript and its scroll position
        self.nodes.update({n.id: n for n in new})
        self.costs = costs if since is None else self.costs | costs
        for trace in changed:
            self.executions.pop(trace, None)
        if first and self.args.conversation:
            self.show_tree(select=self.open_conversation(self.args.conversation))
        else:
            # Someone else's turn (another window, another machine) extends the branch on screen: follow it.
            tip = self.selected.id if self.selected else None
            for node in new:
                if tip is not None and node.parent == tip:
                    tip = node.id
                if not first:
                    self.announce(node)
            self.show_tree(select=tip if self.selected and tip != self.selected.id else None)
            self.select(self.selected)
        self.log_line(f'{"full" if since is None else "poll"}: {len(nodes)} nodes, {len(new)} new')

    def open_conversation(self, wanted: str) -> str | None:
        """The newest node of the conversation `wanted` (a node ID or prefix in it, or `latest`)."""
        roots = sorted((n for n in self.nodes.values() if n.parent not in self.nodes), key=lambda n: n.at)
        if wanted == 'latest':
            root = roots[-1] if roots else None
        else:
            node = next((n for i, n in self.nodes.items() if i.startswith(wanted)), None)
            while node is not None and node.parent in self.nodes:
                node = self.nodes[node.parent]
            root = node
        if root is None:
            self.notify(f'No conversation {wanted!r}', severity='warning')
            return None
        return max(self.subtree(root), key=lambda n: n.at).id

    def conversation_id(self) -> str | None:
        node = self.selected
        while node is not None and node.parent in self.nodes:
            node = self.nodes[node.parent]
        return node.id if node else None

    def copy_conversation_id(self) -> None:
        if root := self.conversation_id():
            self.copy_to_clipboard(root)
            self.notify(f'Copied {root}; open it with --conversation {root[:8]}')

    def add_node(self, node: Node) -> None:
        """A node a running turn just recorded: show it before Logfire can be queried for it."""
        follow = (self.selected.id if self.selected else None) == node.parent
        self.nodes[node.id] = node
        self.show_tree(select=node.id if follow else None)
        self.announce(node)

    def announce(self, node: Node) -> None:
        if node.kind == 'question':
            self.notify(one_line(node.text, 120), title='Agent asks', timeout=20)
        elif node.kind == 'output':
            self.notify(one_line(node.text, 120), title='Done')
        elif node.kind == 'error':
            self.notify(f'{one_line(node.text, 120)}\nctrl+g retries it', title='Failed', severity='error')

    def show_tree(self, select: str | None = None) -> None:
        panel = self.query_one('#conversations', OptionList)
        keep = select or (self.selected.id if self.selected else None)
        nodes = sorted(self.nodes.values(), key=lambda n: n.at)
        children: dict[str | None, list[Node]] = {}
        for node in nodes:
            children.setdefault(node.parent if node.parent in self.nodes else None, []).append(node)
        open_root = self.root_of(keep)  # the conversation holding the selection (archived or not)

        # One row per conversation; the open one unfolds into its branches.
        with self.batch_update():
            roots = children.get(None, [])
            hidden = {r.id for r in roots if self.is_archived(r.id) and r.id != open_root and not self.show_archived}
            conversations = {r.id: self.subtree(r, children) for r in roots if r.id not in hidden}
            self.latest = {root: max(n.at for n in nodes) for root, nodes in conversations.items()}
            if self.seen is None:  # what is there at startup counts as seen; updates after it get the dot
                self.seen = dict(self.latest)
            if open_root in self.latest:  # updates to the conversation on screen are seen as they arrive
                self.seen[open_root] = self.latest[open_root]
            # Most recently updated first, like an inbox.
            order = sorted(conversations, key=lambda r: self.latest[r], reverse=True)
            self.order = order
            self.fill_sidebar(panel, open_root, None if select else keep)
            if select is not None:
                self.select(self.nodes[select])

    def fill_sidebar(self, panel: OptionList, open_root: str | None, selected: str | None) -> None:
        branches = self.branch_tree(self.nodes[open_root]) if open_root in self.nodes else []
        on = self.branch_of(branches, selected)
        options, self.rows, highlight = [], {}, None
        for root in self.order:
            self.rows[root] = len(options)
            options.append(Option(self.row_label(self.nodes[root], self.subtree(self.nodes[root])), id=root))
            if root != open_root:
                continue
            highlight = len(options) - 1
            for branch in branches[1:] if len(branches) > 1 else []:
                if branch is on:
                    highlight = len(options)
                options.append(Option(self.branch_label(branch), id=f'>{branch.start.id}'))
        scroll = panel.scroll_y
        panel.clear_options()
        panel.add_options(options)
        panel.highlighted = highlight
        panel.scroll_to(y=scroll, animate=False)

    def branch_tree(self, root: Node) -> list[Branch]:
        """The conversation cut at its branch points, depth first, oldest branch first."""
        children: dict[str | None, list[Node]] = {}
        for node in sorted(self.subtree(root), key=lambda n: n.at):
            children.setdefault(node.parent, []).append(node)
        found: list[Branch] = []

        def walk(start: Node, depth: int, guides: str, siblings: list[Node]) -> None:
            node = start
            while len(children.get(node.id, [])) == 1:
                node = children[node.id][0]
            below = children.get(node.id, [])
            found.append(Branch(start, self.subtree(start), depth, guides, not below, siblings))
            for index, child in enumerate(below):
                last = index == len(below) - 1
                at = len(found)
                walk(child, depth + 1, guides + ('  ' if last else '│ '), below)
                found[at].guides = guides + ('└ ' if last else '├ ')

        walk(root, 0, '', [root])
        return found

    def branch_of(self, branches: list[Branch], node_id: str | None) -> Branch | None:
        """The deepest branch holding the node."""
        node = self.nodes.get(node_id) if node_id else None
        return max((b for b in branches if node in b.nodes), key=lambda b: b.depth, default=None)

    def branch_label(self, branch: Branch) -> Table:
        """What started the branch, then how it ended and its cost; ⌘ counts its code runs."""
        tip, start = branch.tip, branch.start
        icon, style = ICONS[tip.kind] if tip.kind in STANDING else ('◌', 'yellow')
        if start.kind == 'fork':
            title = Text.assemble((ICONS['fork'][0], ICONS['fork'][1]), f' re-run {start.text.rsplit(":", 1)[-1]}')
        elif start.kind in ('prompt', 'answer'):
            title = label(start)
        else:  # the original line past a fork: named by where it ends
            title, start = label(tip), tip
        runs = sum(len(self.executions.get(t, [])) for t in {n.trace for n in branch.nodes if n.trace})
        row = Table.grid(expand=True)
        row.add_column(no_wrap=True, overflow='ellipsis', ratio=1)
        row.add_column(justify='right', no_wrap=True)
        row.add_row(
            Text.assemble((branch.guides, 'dim'), title),
            Text.assemble((f'⌘{runs} ' if runs else '', 'dim'), (when(tip.at), 'dim'), ' ', (icon, style)),
        )
        if branch.leaf:
            under = branch.guides[:-2] + ('  ' if branch.guides.endswith('└ ') else '│ ') + '  '
            outcome = label(tip).plain if tip is not start else ''
            row.add_row(Text(under + outcome, 'dim'), Text(price(self.cost(branch.nodes)), 'dim'))
        return row

    def action_sibling(self, step: int) -> None:
        """The next or previous branch from the same branch point."""
        root = self.conversation_id()
        if root is None:
            return
        branches = self.branch_tree(self.nodes[root])
        here = self.branch_of(branches, self.selected.id if self.selected else None)
        while here is not None and len(here.siblings) < 2:
            here = self.branch_of(branches, here.start.parent)
        if here is None:
            return
        sibling = here.siblings[(here.siblings.index(here.start) + step) % len(here.siblings)]
        self.select(max(self.subtree(sibling), key=lambda n: n.at))

    def row_label(self, root: Node, nodes: list[Node]) -> Table:
        """Two lines: the first prompt and where the conversation stands; then its cost, and how long ago it changed
        (● when that is since you last looked in this terminal)."""
        latest = self.latest.get(root.id, root.at)
        unread = latest > (self.seen or {}).get(root.id, '')
        newest = max(nodes, key=lambda n: n.at, default=root)
        icon, style = ICONS[newest.kind] if newest.kind in STANDING else ('◌', 'yellow')  # mid-turn
        row = Table.grid(expand=True)
        row.add_column(no_wrap=True, overflow='ellipsis', ratio=1)
        row.add_column(justify='right', no_wrap=True)
        archived = '▣ archived · ' if self.is_archived(root.id) else ''
        parents = {n.parent for n in nodes}
        tips = sum(1 for n in nodes if n.id not in parents)
        forks = f' · {ICONS["fork"][0]}{tips}' if tips > 1 else ''
        row.add_row(Text(one_line(root.text, 80), 'bold' if unread else ''), Text(f' {icon}', style))
        row.add_row(
            Text(f'{archived}{price(self.cost(nodes)) or "$0"}{forks}', 'dim'),
            Text.assemble(('● ' if unread else '', 'bold cyan'), (when(latest), '' if unread else 'dim')),
        )
        return row

    def relabel_row(self, root: str) -> None:
        if root in self.rows and root in self.nodes:
            panel = self.query_one('#conversations', OptionList)
            panel.replace_option_prompt(root, self.row_label(self.nodes[root], self.subtree(self.nodes[root])))

    def relabel(self) -> None:
        for root in self.rows:
            self.relabel_row(root)
        self.show_runs(self.runs_shown)

    def mark_seen(self, node: Node | None) -> None:
        """The conversation on screen is read up to its newest node."""
        root = self.root_of(node.id if node else None)
        if root is None or self.seen is None or root not in self.latest:
            return
        if self.seen.get(root) != self.latest[root]:
            self.seen[root] = self.latest[root]
            self.relabel_row(root)

    def unread(self) -> int:
        return sum(1 for root, at in self.latest.items() if at > (self.seen or {}).get(root, ''))

    def root_of(self, node_id: str | None) -> str | None:
        node = self.nodes.get(node_id) if node_id else None
        while node is not None and node.parent in self.nodes:
            node = self.nodes[node.parent]
        return node.id if node else None

    def branches(self) -> list[Node]:
        """The tips of the selected conversation's branches, newest first."""
        root = self.conversation_id()
        if root is None:
            return []
        nodes = self.subtree(self.nodes[root])
        parents = {n.parent for n in nodes}
        return sorted((n for n in nodes if n.id not in parents), key=lambda n: n.at, reverse=True)

    def show_branch(self, tip: Node) -> None:
        self.select(tip)

    def branch_title(self, tip: Node) -> str:
        """What started the branch (its newest prompt, answer or re-run) and where it ends."""
        node: Node | None = tip
        while node is not None and node.kind not in ('prompt', 'answer', 'fork'):
            node = self.nodes.get(node.parent) if node.parent else None
        start = f'{describe(node)} → ' if node is not None and node is not tip else ''
        return f'{start}{describe(tip)}'

    def subtree(self, root: Node, children: dict[str | None, list[Node]] | None = None) -> list[Node]:
        if children is None:
            children = {}
            for node in self.nodes.values():
                children.setdefault(node.parent, []).append(node)
        nodes, todo = [], [root]
        while todo:
            node = todo.pop()
            nodes.append(node)
            todo.extend(children.get(node.id, []))
        return nodes

    def cost(self, nodes: list[Node]) -> float:
        return sum(self.costs.get(t, 0.0) for t in {n.trace for n in nodes if n.trace})

    @work(group='executions')
    async def load_executions(self, traces: list[str]) -> None:
        """Code runs and model requests of these turns, read from their traces."""
        store = TraceStore(self.sql)
        found: tuple[list[Execution], list[Request]] = ([], [])
        try:
            found = await asyncio.gather(store.executions(traces), store.requests(traces))
        except Exception as error:  # noqa: BLE001 - shown in the log pane; the transcript shows nodes only
            self.log_line(f'trace details query failed: {error}')
        for trace in traces:
            self.executions[trace] = [e for e in found[0] if e.trace == trace]
            self.requests[trace] = [r for r in found[1] if r.trace == trace]
        if self.fork is None:  # re-rendering would drop a pick in progress
            self.select(self.selected)

    @on(OptionList.OptionSelected, '#conversations')
    def on_conversation_clicked(self, event: OptionList.OptionSelected) -> None:
        """A click (or enter) opens the conversation at its newest message, even one already on screen."""
        if (start := self.nodes.get((event.option.id or '').removeprefix('>'))) is not None:
            self.select(max(self.subtree(start), key=lambda n: n.at))
            self.query_one('#composer', Input).focus()

    @on(OptionList.OptionHighlighted, '#conversations')
    def on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        option = event.option.id or ''
        data = self.nodes.get(option.removeprefix('>'))
        # Highlights arrive as messages: one the list has moved on from (a rebuild, a selection made elsewhere) is
        # not a pick.
        if data is None or event.option_index != event.option_list.highlighted:
            return
        if option.startswith('>'):  # a branch row: its newest node, unless the selection is on it already
            if self.selected not in self.subtree(data):
                self.select(max(self.subtree(data), key=lambda n: n.at))
            return
        # A conversation shows its newest branch through the last message; a branch of it already on screen (one
        # you were following, or switched to) stays.
        if self.root_of(self.selected.id if self.selected else None) == data.id:
            return
        self.select(max(self.subtree(data), key=lambda n: n.at))

    def select(self, node: Node | None) -> None:
        self.selected = node
        self.fork = None
        self.mark_seen(node)
        root = self.root_of(node.id if node else None)
        self.fill_sidebar(self.query_one('#conversations', OptionList), root, node.id if node else None)
        self.show_where()
        transcript = self.query_one('#transcript', VerticalScroll)
        path = []
        while node is not None:
            path.append(node)
            node = self.nodes.get(node.parent) if node.parent else None
        path.reverse()
        traces = list(dict.fromkeys(n.trace for n in path if n.trace))
        # Every turn of the conversation, so the code-runs panel lists the other branches too.
        everywhere = {n.trace for n in self.subtree(path[0]) if n.trace} if path else set()
        missing = sorted({*traces, *everywhere} - self.executions.keys())
        if missing:
            self.executions.update({t: [] for t in missing})  # once per trace, until the next refresh
            self.load_executions(missing)
        executions = on_path(path, [e for t in traces for e in self.executions.get(t, [])])
        entries: list[Item] = sorted([*path, *executions], key=lambda e: e.at)
        self.items = entries
        self.step_requests = self.match_requests(path, traces)
        # The newest message stays readable without ctrl+o: the last model step, unless a node follows it (a code
        # run or a failure does not count, so a running or failed turn shows what the model last said).
        tail = [e for e in reversed(entries) if isinstance(e, Node) and e.kind != 'error']
        self.last_step = tail[0].id if tail and tail[0].kind == 'model' else None
        if path:
            ran = f' · {len(executions)} code runs' if executions else ''
            branch, total = price(self.cost(path)), price(self.cost(self.subtree(path[0])))
            costs = f'branch {branch} · conversation {total}' if total else 'no model cost recorded yet'
            tips = [b.tip for b in self.branch_tree(path[0]) if b.leaf]  # in sidebar order
            below = self.subtree(path[-1])
            on = max((i for i, t in enumerate(tips) if t in below), key=lambda i: tips[i].at, default=None)
            which = f'branch {on + 1}/{len(tips)} (← →) · ' if len(tips) > 1 and on is not None else ''
            self.query_one('#cost', Static).update(f'{which}{costs}{ran}')
        else:
            self.query_one('#cost', Static).update('')
        with self.batch_update():
            transcript.remove_children()
            if not path:
                transcript.mount(Static(EMPTY, classes='empty'))
            else:
                transcript.mount_all(self.entry(e) for e in entries)
        transcript.scroll_end(animate=False)
        self.show_runs(self.conversation_runs())

    def match_requests(self, path: list[Node], traces: list[str]) -> dict[str, tuple[Request, bool]]:
        """Each model step's request, and whether a sub-agent made it.

        A step is the checkpoint taken just before a request, so in each turn they pair up in order. The turn's
        first request is the main agent's; requests with other instructions come from its sub-agents.
        """
        matched = {}
        for trace in traces:
            requests = list(self.requests.get(trace, []))
            main = requests[0].instructions if requests else None
            steps = sorted(
                (n for n in self.nodes.values() if n.trace == trace and n.kind == 'model'), key=lambda n: n.at
            )
            for step, request in zip(steps, requests, strict=False):
                matched[step.id] = (request, request.instructions != main)
        return {n.id: matched[n.id] for n in path if n.id in matched}

    def entry(self, item: Item) -> Pickable:
        if isinstance(item, Execution):
            return CodeEntry(item)
        request, sub_agent = self.step_requests.get(item.id, (None, False))
        return Entry(item, request, sub_agent=sub_agent, expanded=self.expanded or item.id == self.last_step)

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        point = self.fork or (self.fork_point(self.selected) if self.selected else ForkPoint(None, '', ''))
        if point is None:
            self.notify('Nothing to continue from here', severity='warning')
            return
        if self.launch(point.target, event.value.strip()):
            event.input.value = ''
            self.unpick()

    def launch(self, node: Node | None, text: str) -> bool:
        if node is not None and not node.resumable:
            self.notify(f'A {node.kind} is not a step to continue from', severity='warning')
            return False
        if not text and (node is None or node.kind != 'model'):
            return False
        if node is not self.selected:
            self.select(node)  # the transcript follows the new branch from where it starts
        self.run_turn(node, text)
        return True

    @work(group='turns')
    async def run_turn(self, node: Node | None, text: str) -> None:
        run_id = id(asyncio.current_task())
        self.runs[run_id] = Run(text or f're-run from {node.id[:8]}' if node else text)
        stack = AsyncExitStack()
        store = TraceStore(self.sql, parent=node.id if node else None, on_record=self.add_node)
        model = text if node is not None and node.kind == 'model' and text else self.args.model
        sandbox = Sandbox(
            model=model,
            workspace=self.args.workspace,
            workspace_mode='memory',
            remote=self.args.remote,
            agent_name=Path(self.args.script).stem,
        )
        self.log_line(f'{node.kind if node else "new"} -> {text or model}')
        # The conversation's ID is its first prompt's node ID: every span of every turn carries it (as Claude's do).
        conversation_id = self.root_of(node.id) if node else uuid4().hex
        stack.enter_context(conversation(conversation_id or uuid4().hex))
        span = logfire.span('tui turn {kind}', kind=node.kind if node else 'new')
        stack.enter_context(span)  # one trace per turn: its nodes, model costs and code executions
        try:
            # The first node goes in before checkout, so a refused checkout still shows under it.
            if node is None:
                await store.record('prompt', text, node_id=conversation_id, script=str(self.args.script), model=model)
            elif node.kind == 'output':
                await store.record('prompt', text, model=model)
            elif node.kind == 'question':
                await store.record('answer', text)
            else:
                await store.record('fork', model, model=model)
            if self.args.files == 'session':
                pool = logfire.agent_sandboxes(api_key=sandbox_key(), url=self.args.remote, auto_resume=False)
                store.files_pool = await stack.enter_async_context(pool)
            async with sandbox.session() as session:
                if node is None:
                    await session.start(self.args.script)
                    output = await session.send(text, store)
                elif node.kind == 'output':
                    await session.load(await store.load(node.id))
                    output = await session.send(text, store)
                else:
                    await session.restore(await store.load(node.id))
                    answer = text if node.kind == 'question' else None
                    output = await session.resume(store, answer=answer)
                if not isinstance(output, Suspended):
                    result = to_jsonable_python(getattr(output, 'attributes', output))
                    text = result if isinstance(result, str) else json.dumps(result, indent=2)
                    await store.save_output(await session.dump(), text)
            await store.node(store.parent)  # wait until the new tip is queryable
        except Exception as error:  # noqa: BLE001 - recorded as an error node in the tree
            await store.record('error', f'{type(error).__name__}: {error}')
            self.log_line(f'error: {type(error).__name__}: {error}')
        finally:
            self.runs.pop(run_id, None)
            await stack.aclose()
        if span.context is not None:
            self.after_turn(store, f'{span.context.trace_id:032x}')

    @work(group='refresh-costs')
    async def after_turn(self, store: TraceStore, trace: str) -> None:
        """The turn's spans are exported by now: fetch its cost, and its code executions on the next render."""
        await asyncio.to_thread(logfire.force_flush)
        await asyncio.sleep(2)
        started = min((n.at for n in self.nodes.values() if n.trace == trace), default=None)
        try:
            since = datetime.fromisoformat(started) if started else None
            self.costs |= await store.costs([trace], since=since)
        except Exception as error:  # noqa: BLE001 - shown in the log pane
            self.log_line(f'cost query failed: {error}')
        self.executions.pop(trace, None)
        self.show_tree()
        self.select(self.selected)


EMPTY = """New conversation

Type a task below and press Enter. The agent plans, delegates to specialists (coder, analyst, writer, critic),
runs code in isolated sandboxes, and asks you when it needs to.

  ctrl+p   commands; anything that matches none runs as a prompt
  ctrl+n   new conversation        ctrl+r   refresh from Logfire
  escape   conversations           ctrl+l   activity log

Every step is a checkpoint in Logfire: pick one in the sidebar to answer, continue, or re-run it."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--remote', default='https://logfire-us.pydantic.dev/', help='Logfire backend URL')
    parser.add_argument('--gateway', help='AI Gateway URL, for a backend without a known one (self-hosted)')
    parser.add_argument('--model', default='gateway/anthropic:claude-haiku-5-5')
    parser.add_argument('--script', type=Path, default=Path(__file__).parent / 'examples' / 'general_agent.py')
    parser.add_argument('--workspace', type=Path, help='seed files for new conversations')
    parser.add_argument('--reload', action='store_true', help='reload the code when a module next to this one changes')
    parser.add_argument(
        '--conversation', help="open this conversation (an ID or prefix of any node in it, or 'latest')"
    )
    parser.add_argument(
        '--poll', type=float, default=5, help='seconds between Logfire queries for new nodes; 0 for never'
    )
    parser.add_argument(
        '--files',
        choices=['span', 'session'],
        default='span',
        help='workspace files on the record as JSON, or in a stored session of their own',
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    def show(url: str, code: str) -> None:
        print(f'Approve this login in your browser: {url}\nCode: {code}', flush=True)

    session = asyncio.run(login(args.remote, show, gateway=args.gateway))
    os.environ.update(session.env())
    export_with(lambda: os.environ['LOGFIRE_TOKEN'])
    configure_logfire()
    app = MontyTui(args, session)
    while (result := app.run()) == RELOAD or (args.reload and app.return_code):
        if result != RELOAD:
            # Keep the login across a crash too: the traceback is above; fix the code and save.
            print('the app crashed; waiting for the next change', flush=True)
            next_change()
        module = reloaded()
        args = module.parse_args()  # new options and defaults take effect too
        app = module.MontyTui(args, app.login)


def reloaded() -> Any:
    """The `tui` module with fresh code; on an error, waits for the next save and tries again."""
    while True:
        try:
            for name in MODULES:
                module = sys.modules.get(name)
                module = importlib.reload(module) if module else importlib.import_module(name)
            return module
        except Exception as error:  # noqa: BLE001 - fix the code and save again
            print(f'reload failed: {type(error).__name__}: {error}\nwaiting for the next change', flush=True)
            next_change()


def next_change() -> None:
    from watchfiles import PythonFilter, watch

    next(watch(HERE, watch_filter=PythonFilter(), recursive=False))


if __name__ == '__main__':
    main()
