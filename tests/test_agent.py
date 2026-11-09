from pathlib import Path

import pytest
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_monty import MontyRuntimeError

from host import Sandbox, State, StateStore, run, run_once

EXAMPLES = Path(__file__).parents[1] / 'examples'
NOTES_AGENT = EXAMPLES / 'notes_agent.py'
LOG_AGENT = Path(__file__).parent / 'agents' / 'log_agent.py'

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / 'workspace'
    path.mkdir()
    (path / 'todo.txt').write_text('buy milk')
    return path


def appender(label: str, calls: int = 2) -> FunctionModel:
    """Per turn, call `append(f'{label}{n}')` until `calls` tool returns exist, then answer with the last return."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [p for m in messages for p in m.parts]
        turn = max(i for i, p in enumerate(parts) if p.part_kind == 'user-prompt')
        returns = [p.content for p in parts[turn:] if p.part_kind == 'tool-return']
        if len(returns) < calls:
            return ModelResponse(parts=[ToolCallPart('append', {'line': f'{label}{len(returns) + 1}'})])
        return ModelResponse(parts=[TextPart(f'done {returns[-1] if returns else 0}')])

    return FunctionModel(respond)


def test_structured_output_with_async_tool(workspace: Path):
    output = run(NOTES_AGENT, prompt='hi', model='test', workspace=workspace, workspace_mode='memory')
    assert output.attributes['files'] == ['a']
    assert sorted(p.name for p in workspace.iterdir()) == ['todo.txt']


def test_output_validation_retry(workspace: Path):
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        files = 'todo.txt' if len(seen) == 1 else ['todo.txt']
        return ModelResponse(parts=[ToolCallPart('final_result', {'summary': 'milk', 'files': files})])

    output = run(NOTES_AGENT, prompt='hi', model=FunctionModel(respond), workspace=workspace)
    assert output.attributes == {'summary': 'milk', 'files': ['todo.txt']}
    retried = seen[1][-1].parts[0]
    assert (retried.outcome, retried.retry.cause) == ('retried', 'validation_error')
    assert 'files: expected list' in retried.content


def test_unknown_tool_retry_then_write_through(workspace: Path):
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        step = len(seen)
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('nope', {})])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart('write_note', {'name': 'new.txt', 'content': 'hello'})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'summary': 'ok', 'files': ['new.txt']})])

    sandbox = {'workspace': workspace, 'workspace_mode': 'read-write', 'env': {'USER': 'ada'}}
    output = run(NOTES_AGENT, prompt='write', model=FunctionModel(respond), **sandbox)
    assert output.attributes['summary'] == 'ok'
    assert seen[1][-1].parts[0].retry.cause == 'unknown_tool'
    assert (workspace / 'new.txt').read_text() == 'hello\n-- ada\n'


def test_host_filesystem_is_unreachable(tmp_path: Path):
    script = tmp_path / 'escape.py'
    script.write_text("open('/etc/passwd').read()")
    with pytest.raises(MontyRuntimeError, match='PermissionError'):
        run(script, prompt='')


async def test_history_is_pydantic_ai_compatible():
    sandbox = Sandbox(model=appender('x', calls=1), workspace_mode='memory')
    async with sandbox.session() as session:
        await session.start(LOG_AGENT)
        assert await session.send('go') == 'done 1'
        history = ModelMessagesTypeAdapter.validate_python(await session.history())
    assert [m.kind for m in history] == ['request', 'response', 'request', 'response']

    native = await PydanticAgent(appender('y', calls=0)).run('hello')
    async with sandbox.session() as session:
        await session.start(LOG_AGENT)
        await session.set_history(ModelMessagesTypeAdapter.dump_python(native.all_messages(), mode='json'))
        await session.send('again')
        history = ModelMessagesTypeAdapter.validate_python(await session.history())
    assert [p.content for m in history for p in m.parts if p.part_kind == 'user-prompt'] == ['hello', 'again']


async def checkpointed_run(tmp_path: Path, mode: str, workspace: Path) -> StateStore:
    store = StateStore(tmp_path / f'checkpoints-{mode}')
    sandbox = Sandbox(model=appender('step'), workspace=workspace, workspace_mode=mode)
    assert await run_once(sandbox, LOG_AGENT, 'go', store) == 'done 2'
    assert len(store.paths()) == 3
    return store


async def resume(sandbox: Sandbox, checkpoint: State) -> tuple[str, dict | None]:
    async with sandbox.session() as session:
        await session.restore(checkpoint)
        return await session.resume(), session.files()


async def test_rollback_memory_workspace_restores_files(tmp_path: Path, workspace: Path):
    store = await checkpointed_run(tmp_path, 'memory', workspace)
    first, middle, last = (store.load(p) for p in store.paths())
    assert '/workspace/log.txt' not in first.files
    assert middle.files['/workspace/log.txt'] == 'step1\n'

    sandbox = Sandbox(model=appender('step'), workspace=workspace, workspace_mode='memory')
    for checkpoint in (first, middle, last):
        output, files = await resume(sandbox, checkpoint)
        assert (output, files['/workspace/log.txt']) == ('done 2', 'step1\nstep2\n')
    assert not (workspace / 'log.txt').exists()


async def test_fork_is_independent(tmp_path: Path, workspace: Path):
    store = await checkpointed_run(tmp_path, 'memory', workspace)
    middle = store.load(store.paths()[1])
    forked = await resume(Sandbox(model=appender('fork', 3), workspace_mode='memory'), middle)
    original = await resume(Sandbox(model=appender('step'), workspace_mode='memory'), middle)
    assert forked == ('done 3', {'/workspace/todo.txt': 'buy milk', '/workspace/log.txt': 'step1\nfork2\nfork3\n'})
    assert original[1]['/workspace/log.txt'] == 'step1\nstep2\n'


async def test_rollback_read_write_replays_side_effects(tmp_path: Path, workspace: Path):
    store = await checkpointed_run(tmp_path, 'read-write', workspace)
    assert store.load(store.paths()[1]).files is None
    sandbox = Sandbox(model=appender('step'), workspace=workspace, workspace_mode='read-write')
    assert await resume(sandbox, store.load(store.paths()[1])) == ('done 3', None)
    assert (workspace / 'log.txt').read_text() == 'step1\nstep2\nstep2\n'


async def test_rollback_overlay_loses_earlier_writes(tmp_path: Path, workspace: Path):
    store = await checkpointed_run(tmp_path, 'overlay', workspace)
    sandbox = Sandbox(model=appender('step'), workspace=workspace, workspace_mode='overlay')
    assert await resume(sandbox, store.load(store.paths()[1])) == ('done 1', None)


async def test_idle_state_rollback_between_turns(tmp_path: Path, workspace: Path):
    store = StateStore(tmp_path / 'states')
    sandbox = Sandbox(model=appender('t', calls=1), workspace=workspace, workspace_mode='memory')
    async with sandbox.session() as session:
        await session.start(LOG_AGENT)
        await session.send('one')
        after_one = store.save(await session.dump())
        await session.send('two')
        assert len(await session.history()) == 8

    async with sandbox.session() as session:
        await session.load(store.load(after_one))
        assert len(await session.history()) == 4
        await session.send('two, again')
        prompts = [p['content'] for m in await session.history() for p in m['parts'] if p['part_kind'] == 'user-prompt']
        assert prompts == ['one', 'two, again']
        assert session.files()['/workspace/log.txt'] == 't1\nt1\n'


def test_tampered_state_is_rejected(tmp_path: Path):
    store = StateStore(tmp_path, key=b'secret')
    path = store.save(State(b'MONTY\0dump', {'/workspace/a': 'x'}))
    assert store.load(path).files == {'/workspace/a': 'x'}
    path.write_bytes(b'MONTY\0evil')
    with pytest.raises(ValueError, match='HMAC'):
        store.load(path)
