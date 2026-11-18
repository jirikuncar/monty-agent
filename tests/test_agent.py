import json
from pathlib import Path

import logfire
import pytest
from logfire.testing import CaptureLogfire
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_monty import MontyRuntimeError

from host import AgentSession, Sandbox, State, StateStore, Suspended, conversation, run, run_once

EXAMPLES = Path(__file__).parents[1] / 'examples'
NOTES_AGENT = EXAMPLES / 'notes_agent.py'
LOG_AGENT = Path(__file__).parent / 'agents' / 'log_agent.py'
CODE_AGENT = EXAMPLES / 'code_agent.py'
ASK_AGENT = EXAMPLES / 'ask_agent.py'

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


def test_child_sandbox_runs_code_isolated(tmp_path: Path):
    programs = ['1 / 0', "open('/app/pydantic_ai_lite.py').read()", 'print("hi")\nsum(range(10))']

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p.content for m in messages for p in m.parts if p.part_kind == 'tool-return']
        if len(returns) < len(programs):
            return ModelResponse(parts=[ToolCallPart('run_python', {'code': programs[len(returns)]})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': repr(returns), 'code': programs[-1]})])

    output = run(CODE_AGENT, prompt='go', model=FunctionModel(respond)).attributes
    zero, isolated, ok = eval(output['answer'])
    assert 'ZeroDivisionError' in zero['error']
    assert 'error' in isolated  # the parent's /app mount is not visible to the child
    assert ok == {'output': '45', 'stdout': 'hi\n'}


def test_spans_follow_genai_conversation_conventions(capfire: CaptureLogfire):
    """Like Claude Code's: the turn is `invoke_agent`, host functions are `execute_tool`, and every span of the
    conversation, model requests included, carries its `gen_ai.conversation.id`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(p.part_kind == 'tool-return' for m in messages for p in m.parts):
            return ModelResponse(parts=[ToolCallPart('run_python', {'code': '1 + 1'})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': '2', 'code': '1 + 1'})])

    logfire.instrument_pydantic_ai()
    try:
        with conversation('c0ffee'):
            run(CODE_AGENT, prompt='go', model=FunctionModel(respond))
    finally:
        PydanticAgent.instrument_all(False)
    spans = {s['name']: s['attributes'] for s in capfire.exporter.exported_spans_as_dict()}
    assert spans['agent turn {prompt}']['gen_ai.operation.name'] == 'invoke_agent'
    assert spans['agent turn {prompt}']['gen_ai.agent.name'] == 'code_agent'  # what makes it an agent run
    turn = {k: json.loads(v) for k, v in spans['agent turn {prompt}'].items() if k.endswith('.messages')}
    assert turn['gen_ai.input.messages'] == [{'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}]
    [reply] = turn['gen_ai.output.messages']  # Logfire shows a run's transcript from these two
    assert reply['role'] == 'assistant' and json.loads(reply['parts'][0]['content'])['answer'] == '2'
    assert spans['child sandbox']['gen_ai.operation.name'] == 'execute_tool'
    assert spans['child sandbox']['gen_ai.tool.name'] == 'run_python'
    assert any(name.startswith('chat ') for name in spans)
    assert {a.get('gen_ai.conversation.id') for a in spans.values()} == {'c0ffee'}


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


async def test_suspend_then_restart(tmp_path: Path, workspace: Path):
    store = StateStore(tmp_path / 'steps')
    sandbox = Sandbox(model=appender('step'), workspace=workspace, workspace_mode='memory')
    async with sandbox.session() as session:
        await session.start(LOG_AGENT)
        suspended = await session.send('go', store, suspend_at=2)
    assert suspended == Suspended(store.paths()[1])
    assert len(store.paths()) == 2
    assert await resume(sandbox, store.load(suspended.checkpoint)) == (
        'done 2',
        {'/workspace/todo.txt': 'buy milk', '/workspace/log.txt': 'step1\nstep2\n'},
    )


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


def asker() -> FunctionModel:
    """Ask once when the prompt lacks a time, then save the invitation with the answer."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [p for m in messages for p in m.parts]
        answers = [p.content for p in parts if p.part_kind == 'tool-return' and p.tool_name == 'ask_user']
        saved = [p.content for p in parts if p.part_kind == 'tool-return' and p.tool_name == 'write_note']
        if not answers:
            return ModelResponse(parts=[ToolCallPart('ask_user', {'question': 'When?'})])
        if not saved:
            return ModelResponse(parts=[ToolCallPart('write_note', {'name': 'invite.md', 'text': answers[0]})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'path': saved[0], 'text': answers[0]})])

    return FunctionModel(respond)


async def test_ask_user_answered_by_host(workspace: Path):
    questions = []

    def ask(question: str) -> str:
        questions.append(question)
        return 'Friday 10:00'

    sandbox = Sandbox(model=asker(), workspace=workspace, workspace_mode='memory', ask=ask)
    output = await run_once(sandbox, ASK_AGENT, 'invite the team')
    assert questions == ['When?']
    assert output.attributes == {'path': '/workspace/invite.md', 'text': 'Friday 10:00'}


async def test_ask_user_suspends_then_resumes_with_answer(tmp_path: Path, workspace: Path):
    store = StateStore(tmp_path / 'steps')
    sandbox = Sandbox(model=asker(), workspace=workspace, workspace_mode='memory')
    async with sandbox.session() as session:
        await session.start(ASK_AGENT)
        suspended = await session.send('invite the team', store)
    assert suspended == Suspended(store.paths()[-1], 'When?')

    # The checkpoint survives being answered: answering it again changes the answer in a fresh fork.
    for answer in ['Friday 10:00', 'Monday 15:00']:
        async with sandbox.session() as session:
            await session.restore(store.load(suspended.checkpoint))
            output = await session.resume(answer=answer)
            assert output.attributes == {'path': '/workspace/invite.md', 'text': answer}
            assert session.files() == {'/workspace/todo.txt': 'buy milk', '/workspace/invite.md': answer}


def test_tampered_state_is_rejected(tmp_path: Path):
    store = StateStore(tmp_path, key=b'secret')
    path = store.save(State(b'MONTY\0dump', {'/workspace/a': 'x'}))
    assert store.load(path).files == {'/workspace/a': 'x'}
    path.write_bytes(b'MONTY\0evil')
    with pytest.raises(ValueError, match='HMAC'):
        store.load(path)


GENERAL_AGENT = EXAMPLES / 'general_agent.py'


def test_general_agent_delegates_and_runs_code(workspace: Path):
    """The orchestrator delegates to the coder, which counts words in a workspace file in code mode."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p.content for m in messages for p in m.parts if p.part_kind == 'tool-return']
        if 'delegate' in {t.name for t in info.function_tools}:
            if not returns:
                return ModelResponse(parts=[ToolCallPart('delegate', {'specialist': 'coder', 'task': 'count words'})])
            return ModelResponse(parts=[TextPart(f'orchestrator: {returns[0]}')])
        if not returns:
            code = "len(files['todo.txt'].split())"
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code, 'files': ['todo.txt']})])
        return ModelResponse(parts=[TextPart(f'coder: {returns[0]["output"]}')])

    output = run(GENERAL_AGENT, prompt='go', model=FunctionModel(respond), workspace=workspace, workspace_mode='memory')
    assert output == 'orchestrator: coder: 2'


def test_general_agent_code_mode_without_files():
    """`files` is optional: models often leave it out (a staging trace failed with a TypeError before)."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p.content for m in messages for p in m.parts if p.part_kind == 'tool-return']
        if not returns:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'sum(range(10))'})])
        return ModelResponse(parts=[TextPart(str(returns[0]))])

    output = run(GENERAL_AGENT, prompt='go', model=FunctionModel(respond))
    assert output == "{'output': '45', 'stdout': ''}"


def test_sandbox_pool_uses_the_current_key(monkeypatch: pytest.MonkeyPatch):
    """A refreshed token reaches new pools: `logfire.agent_sandboxes()` would otherwise use the startup key."""
    calls: list[dict] = []
    monkeypatch.setattr(logfire, 'agent_sandboxes', lambda **kwargs: calls.append(kwargs))
    session = AgentSession(Sandbox(remote='https://logfire.example'))
    for token in ('first', 'refreshed'):
        monkeypatch.setenv('LOGFIRE_API_KEY', token)
        session._pool()
    assert [c['api_key'] for c in calls] == ['first', 'refreshed']
