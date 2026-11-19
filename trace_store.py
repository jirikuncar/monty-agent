"""Checkpoints as Logfire records, so any process that can query the project finds and restores them.

A remote session server that stores sessions answers `dump()` with an ID, so a checkpoint is that ID plus a little
metadata. Every record is a node with a parent: prompts, answers and forks start branches, model requests and
questions are checkpoints, outputs end a branch. The tree of nodes is the whole conversation, with no local state.
"""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import httpx
import logfire
from opentelemetry import trace

from host import HOST_SERVICE, State

Query = Callable[[str, dict[str, str], datetime | None], Awaitable[list[dict[str, Any]]]]
"""`(sql, params, since)` -> rows; `params` binds `$name` placeholders to SQL literals, and `since` narrows the
time window to records that start at or after it (None: as far back as the backend allows)."""
Kind = Literal['prompt', 'answer', 'fork', 'model', 'question', 'output', 'error']
PREFIX = 'monty_agent.'
LIST_SQL = f"""\
SELECT trace_id, span_id, start_timestamp, attributes FROM records
WHERE attributes->>'{PREFIX}node' IS NOT NULL{{where}}
ORDER BY start_timestamp
"""
# Model cost per turn trace. The Gateway may report the same requests under its own service, so take the largest
# per-service total rather than the sum.
COST_SQL = f"""\
WITH turns AS (SELECT DISTINCT trace_id FROM records WHERE attributes->>'{PREFIX}node' IS NOT NULL{{traces}}),
per_service AS (
  SELECT r.trace_id, r.service_name, sum((r.attributes->>'operation.cost')::float) AS cost
  FROM records r JOIN turns t ON r.trace_id = t.trace_id
  WHERE r.attributes->>'operation.cost' IS NOT NULL
  GROUP BY r.trace_id, r.service_name
)
SELECT trace_id, max(cost) AS cost FROM per_service GROUP BY trace_id
"""
EXECUTIONS_SQL = """\
SELECT trace_id, span_id, start_timestamp, attributes->>'code' AS code, attributes->>'result' AS result FROM records
WHERE span_name = 'child sandbox' AND trace_id IN ({traces})
ORDER BY start_timestamp
"""
# Each model request as the host's Pydantic AI instrumentation recorded it (the Gateway records its own copy). The
# input repeats the whole history, so only its newest message is read: the prompt, task or tool results it adds.
REQUESTS_SQL = f"""\
SELECT trace_id, span_id, start_timestamp, attributes->>'gen_ai.system_instructions' AS instructions,
  json_get_json(
    attributes->'gen_ai.input.messages', CAST(json_length(attributes->'gen_ai.input.messages') - 1 AS BIGINT)
  ) AS input,
  attributes->>'gen_ai.output.messages' AS output
FROM records
WHERE span_name LIKE 'chat %' AND service_name = '{HOST_SERVICE}' AND trace_id IN ({{traces}})
ORDER BY start_timestamp
"""
# Archiving appends an event naming a conversation's root node; the newest event per root wins. Records are
# immutable, so this is how a conversation leaves the list (it still ages out with the project's retention).
ARCHIVE_SQL = f"""\
SELECT start_timestamp, attributes->>'{PREFIX}archive' AS root, attributes->>'{PREFIX}archived' AS archived
FROM records WHERE attributes->>'{PREFIX}archive' IS NOT NULL
ORDER BY start_timestamp
"""
TRACE_ID = re.compile('[0-9a-f]{32}')


@dataclass
class Node:
    id: str
    parent: str | None
    kind: Kind
    text: str
    at: str
    attrs: dict[str, Any] = field(default_factory=dict)
    trace: str | None = None
    span: str | None = None
    """The turn's trace: its model requests (and their cost) and code executions."""

    @property
    def resumable(self) -> bool:
        """A mid-turn checkpoint (`restore`) or, for `output`, the idle session after the turn (`load`)."""
        return self.kind in ('model', 'question', 'output')


class TraceStore:
    """`StateStore` for remote sessions, kept in Logfire: `save` records a node, `load` finds it by query.

    Workspace files go on the record as JSON, or with `files_pool` into a stored session of their own (one per
    distinct workspace), which keeps records small however big the files get.
    """

    def __init__(
        self,
        query: Query,
        *,
        parent: str | None = None,
        files_pool: Any = None,
        visible_within: float = 30,
        on_record: Callable[[Node], None] | None = None,
    ):
        self.query = query
        self.parent = parent
        self.on_record = on_record
        self.files_pool = files_pool
        self.visible_within = visible_within
        self._files_sessions: dict[str, str] = {}

    async def record(self, kind: Kind, text: str = '', *, node_id: str | None = None, **attrs: Any) -> str:
        """Append a node under the current parent; it becomes the parent of the next one."""
        node_id = node_id or uuid4().hex
        fields = {'node': node_id, 'parent': self.parent, 'kind': kind, 'text': text, **attrs}
        logfire.info(f'monty-agent {kind}', **{PREFIX + k: v for k, v in fields.items() if v is not None})
        await asyncio.to_thread(logfire.force_flush)
        if self.on_record is not None:
            own = {k: v for k, v in attrs.items() if v is not None}
            context = trace.get_current_span().get_span_context()
            trace_id = f'{context.trace_id:032x}' if context.is_valid else None
            self.on_record(Node(node_id, self.parent, kind, text, iso(datetime.now(UTC)), own, trace_id))
        self.parent = node_id
        return node_id

    async def save(self, state: State, meta: dict[str, Any]) -> str:
        """A checkpoint before a model request, or at a question (`meta['question']`)."""
        question = meta.get('question')
        kind: Kind = 'question' if question is not None else 'model'
        return await self._save(kind, question or '', state, messages=meta.get('messages'))

    async def save_output(self, state: State, output: str) -> str:
        """The idle session after a finished turn: `load` it and `send` the next prompt to continue."""
        return await self._save('output', output, state)

    async def _save(self, kind: Kind, text: str, state: State, **attrs: Any) -> str:
        if len(state.monty) != 16:
            raise ValueError('TraceStore keeps session IDs, so it needs a server that stores sessions (--remote)')
        if state.files is not None and self.files_pool is not None:
            attrs['files_session'] = await self._put_files(state)
        elif state.files is not None:
            attrs['files'] = json.dumps(state.encode_files())
        return await self.record(kind, text, node_id=state.monty.hex(), **attrs)

    async def load(self, node_id: str) -> State:
        node = await self.node(node_id)
        assert node.resumable, f'{node.kind} node {node_id} is not a checkpoint'
        if 'files_session' in node.attrs:
            files = await self._get_files(node.attrs['files_session'])
        else:
            files = State.decode_files(parsed(node.attrs.get('files'), None))
        return State(bytes.fromhex(node_id), files)

    async def node(self, node_id: str) -> Node:
        """Wait for a just-written node to become queryable."""
        deadline = time.monotonic() + self.visible_within
        while True:
            found = await self.nodes(f"attributes->>'{PREFIX}node' = $node", {'node': f"'{int(node_id, 16):032x}'"})
            if found:
                return found[0]
            if time.monotonic() > deadline:
                raise LookupError(f'no node {node_id} in Logfire')
            await asyncio.sleep(1)

    async def nodes(
        self, where: str | None = None, params: dict[str, str] | None = None, *, since: datetime | None = None
    ) -> list[Node]:
        rows = await self.query(LIST_SQL.format(where=f' AND {where}' if where else ''), params or {}, since)
        return [to_node(row) for row in rows]

    async def costs(self, traces: Iterable[str] | None = None, *, since: datetime | None = None) -> dict[str, float]:
        """Model cost (USD) per turn trace, for every trace with nodes."""
        if traces is None:
            sql = COST_SQL.format(traces='')
        else:
            traces = [t for t in traces if TRACE_ID.fullmatch(t)]
            if not traces:
                return {}
            sql = COST_SQL.format(traces=f' AND trace_id IN ({", ".join(repr(t) for t in traces)})')
        # A trace with no cost yet gets 0, so a partial earlier total is replaced, not kept.
        found = {row['trace_id']: row['cost'] for row in await self.query(sql, {}, since)}
        return found if traces is None else {t: found.get(t, 0.0) for t in traces}

    async def executions(self, traces: list[str]) -> list[Execution]:
        """The code each turn ran in child sandboxes, oldest first."""
        traces = [t for t in traces if TRACE_ID.fullmatch(t)]
        if not traces:
            return []
        rows = await self.query(EXECUTIONS_SQL.format(traces=', '.join(f"'{t}'" for t in traces)), {}, None)
        return [
            Execution(
                row['trace_id'], iso(row['start_timestamp']), row['code'] or '', row['result'] or '', row.get('span_id')
            )
            for row in rows
        ]

    async def archive(self, root: str, archived: bool = True) -> None:
        """Hide the conversation starting at `root` from lists (or show it again)."""
        logfire.info(
            'monty-agent {action} {root}',
            action='archive' if archived else 'unarchive',
            root=root,
            **{f'{PREFIX}archive': root, f'{PREFIX}archived': archived},
        )
        await asyncio.to_thread(logfire.force_flush)

    async def archived(self, *, since: datetime | None = None) -> dict[str, tuple[str, bool]]:
        """Root node -> (when, archived) of the newest archive event per conversation."""
        events = {}
        for row in await self.query(ARCHIVE_SQL, {}, since):
            value = row['archived']
            events[row['root']] = (iso(row['start_timestamp']), value in (True, 'true', 'True', '1'))
        return events

    async def requests(self, traces: list[str]) -> list[Request]:
        """The model requests each turn made, oldest first: the main agent's and its sub-agents'."""
        traces = [t for t in traces if TRACE_ID.fullmatch(t)]
        if not traces:
            return []
        rows = await self.query(REQUESTS_SQL.format(traces=', '.join(f"'{t}'" for t in traces)), {}, None)
        return [Request.from_row(row) for row in rows]

    async def _put_files(self, state: State) -> str:
        key = hashlib.sha256(json.dumps(state.encode_files(), sort_keys=True).encode()).hexdigest()
        if key not in self._files_sessions:
            async with self.files_pool.checkout(script_name='files.py', ephemeral=False) as session:
                await session.feed_run('stored = files', inputs={'files': state.files})
                self._files_sessions[key] = (await session.dump()).hex()
        return self._files_sessions[key]

    async def _get_files(self, ref: str) -> dict[str, str | bytes]:
        async with self.files_pool.checkout(script_name='files.py', ephemeral=True) as session:
            await session.load_session(bytes.fromhex(ref))
            return await session.feed_run('stored')


def to_node(row: dict[str, Any]) -> Node:
    attrs = row['attributes']
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    own = {k.removeprefix(PREFIX): v for k, v in attrs.items() if k.startswith(PREFIX)}
    return Node(
        id=own.pop('node'),
        parent=own.pop('parent', None),
        kind=own.pop('kind'),
        text=own.pop('text', ''),
        at=iso(row['start_timestamp']),
        attrs=own,
        trace=row.get('trace_id'),
        span=row.get('span_id'),
    )


@dataclass
class Execution:
    """A `child sandbox` span: code the agent ran, and what came back."""

    trace: str
    at: str
    code: str
    result: str
    span: str | None = None


@dataclass
class Request:
    """One model request: which agent made it (its instructions), what was new in its input, and the reply."""

    trace: str
    at: str
    instructions: str
    input: list[dict[str, Any]]
    output: list[dict[str, Any]]
    span: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Request:
        instructions = parsed(row['instructions'], [])
        text = ' '.join(i.get('content', '') for i in instructions if isinstance(i, dict))
        newest = parsed(row['input'], {})
        output = parsed(row['output'], [])
        return cls(
            trace=row['trace_id'],
            at=iso(row['start_timestamp']),
            instructions=text,
            input=newest.get('parts', []) if newest.get('role') == 'user' else [],
            output=[part for message in output for part in message.get('parts', [])],
            span=row.get('span_id'),
        )


def parsed(value: Any, default: Any) -> Any:
    """A JSON column: text from `->>`, but a `json_get_json` result can arrive already parsed."""
    if value is None or value == '':
        return default
    return json.loads(value) if isinstance(value, str | bytes) else value


def iso(at: datetime | str) -> str:
    """One timestamp format for nodes recorded here and rows read back, so they sort together."""
    moment = datetime.fromisoformat(at) if isinstance(at, str) else at
    return moment.astimezone(UTC).isoformat(' ', 'microseconds')


class NotGranted(Exception):
    """The token lacks the scope the MCP tool needs, so the server does not list the tool."""


class McpQuery:
    """Run SQL through the Logfire MCP server's `query_run`: the query API takes read tokens, MCP takes OAuth."""

    def __init__(self, backend: str, access_token: str, *, days: int = 14):
        self.url = f'{backend.rstrip("/")}/mcp'
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        self.days = days
        self._http = httpx.AsyncClient(timeout=60)
        self._ids = 0
        self._tools: set[str] | None = None
        """Tools the token's scopes allow, as the server lists them."""

    def set_token(self, access_token: str) -> None:
        """Use a refreshed token; the MCP session belongs to the old one, so the next call starts another."""
        self.headers['Authorization'] = f'Bearer {access_token}'
        self.headers.pop('mcp-session-id', None)
        self._tools = None  # a new login may grant more

    async def __call__(self, sql: str, params: dict[str, str], since: datetime | None = None) -> list[dict[str, Any]]:
        if 'mcp-session-id' not in self.headers:
            await self._initialize()
        start = datetime.now(UTC) - timedelta(days=self.days) + timedelta(minutes=1)  # `query_run` allows 14 days
        if since is not None:
            start = max(start, since)
        arguments = {'query': sql, 'start_timestamp': start.isoformat(), 'params': params or None}
        result = await self._rpc('tools/call', {'name': 'query_run', 'arguments': arguments})
        text = result['content'][0]['text']
        if result.get('isError'):
            raise RuntimeError(f'query_run failed: {text[:500]}')
        return json.loads(text)['rows']

    async def project_url(self) -> str:
        """The project's Logfire UI page, from the MCP server, which knows the organization and project names."""
        if 'mcp-session-id' not in self.headers:
            await self._initialize()
        result = await self._rpc('tools/call', {'name': 'project_logfire_ui_link', 'arguments': {}})
        text = result['content'][0]['text']
        if result.get('isError'):
            raise RuntimeError(f'project_logfire_ui_link failed: {text[:500]}')
        url = next(word for word in text.replace('"', ' ').split() if word.startswith('http'))
        return url.split('?')[0].rstrip('/')

    async def whoami(self) -> str | None:
        """The token's user, by email; None for a service token."""
        text, error = await self._tool('token_info', {}, scope='project:read')
        if error:
            raise RuntimeError(f'token_info failed: {text[:500]}')
        info = json.loads(text)
        return info.get('user_email') or info.get('user_name')

    async def read_variable(self, name: str, label: str) -> Any:
        """The value under the managed variable's label; None when the variable or the label does not exist yet."""
        text, error = await self._tool('variable_get', {'name': name}, scope='project:read_variables')
        if error:
            if 'not found' in text.lower():
                return None
            raise RuntimeError(f'variable_get failed: {text[:500]}')
        held = json.loads(text)['definition'].get('labels', {}).get(label) or {}
        return parsed(held.get('serialized_value'), None)

    async def write_variable(self, name: str, label: str, value: Any, description: str) -> None:
        """A new version of the managed variable, under the label (which moves off its old version); the first write
        creates the variable."""
        scope = 'project:write_variables'
        version = {'name': name, 'action': 'create_version', 'version': {'value': value, 'labels': [label]}}
        text, error = await self._tool('variable_manage', version, scope=scope)
        if error and 'not found' in text.lower():
            metadata = {'description': description, 'json_schema': {'type': 'object'}}
            text, error = await self._tool(
                'variable_manage', {'name': name, 'action': 'create', 'metadata': metadata}, scope=scope
            )
            if not error:
                text, error = await self._tool('variable_manage', version, scope=scope)
        if error:
            raise RuntimeError(f'variable_manage failed: {text[:500]}')

    async def _tool(self, name: str, arguments: dict[str, Any], *, scope: str) -> tuple[str, bool]:
        if 'mcp-session-id' not in self.headers:
            await self._initialize()
        if self._tools is None:
            self._tools = {tool['name'] for tool in (await self._rpc('tools/list', {}))['tools']}
        if name not in self._tools:
            raise NotGranted(f'log in again to grant {scope}')
        result = await self._rpc('tools/call', {'name': name, 'arguments': arguments})
        return result['content'][0]['text'], bool(result.get('isError'))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _initialize(self) -> None:
        info = {'name': 'monty-agent', 'version': '0'}
        await self._rpc('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': info})
        await self._http.post(
            self.url, headers=self.headers, json={'jsonrpc': '2.0', 'method': 'notifications/initialized'}
        )

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ids += 1
        message = {'jsonrpc': '2.0', 'id': self._ids, 'method': method, 'params': params}
        response = (await self._http.post(self.url, headers=self.headers, json=message)).raise_for_status()
        if session_id := response.headers.get('mcp-session-id'):
            self.headers['mcp-session-id'] = session_id
        body = response.text
        if 'event-stream' in response.headers.get('content-type', ''):
            body = ''.join(line[5:] for line in body.splitlines() if line.startswith('data:'))
        reply = json.loads(body)
        if 'error' in reply:
            raise RuntimeError(f'MCP {method} failed: {reply["error"]}')
        return reply['result']
