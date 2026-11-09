"""Run an agent script inside Monty; every external call is answered here.

The script defines `agent` (and optionally `deps`). Each turn runs `await agent.run(prompt, ...)` in the same session,
so the conversation lives in sandbox globals and can be dumped between turns or checkpointed at every model request.
"""

import argparse
import asyncio
import base64
import contextvars
import hashlib
import hmac
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

import logfire
from pydantic_ai.direct import model_request as direct_model_request
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python
from pydantic_monty import (
    NOT_HANDLED,
    AsyncFunctionSnapshot,
    AsyncMonty,
    AsyncMontySession,
    CollectString,
    ExternalModule,
    MemoryFile,
    MontyComplete,
    MontyError,
    MountDir,
    OSAccess,
    ResourceLimits,
)

SANDBOX_DIR = Path(__file__).parent / 'sandbox'
WORKSPACE = PurePosixPath('/workspace')
MODEL_REQUEST = ('host.model_request', 'host.model_request_async')
SETUP = 'history = []\ndeps = None\n'
TURN = """\
from pydantic_ai_lite import dump_messages
result = await agent.run(prompt, deps=deps, message_history=history)
history = result.all_messages()
result.output
"""

# A child sandbox gets less than its parent: it runs model-written code with nothing to reach.
CHILD_LIMITS = ResourceLimits(max_memory=50_000_000, max_feed_duration_secs=10)
CHILD_OUTPUT_BYTES = 64_000

WorkspaceMode = Literal['overlay', 'read-write', 'memory']
Files = dict[str, str | bytes]


def configure_logfire() -> None:
    """Export host spans and model requests when LOGFIRE_TOKEN is set; sandbox sessions join the same trace."""
    logfire.configure(send_to_logfire='if-token-present', service_name='monty-agent-host', console=False)
    logfire.instrument_pydantic_ai()


def log(msg: str) -> None:
    print(f'[host] {msg}', file=sys.stderr)


def resolve_model(model: Model | str) -> Model:
    if isinstance(model, Model):
        return model
    return TestModel() if model == 'test' else infer_model(model)


def request_parameters(params: dict) -> ModelRequestParameters:
    tools = {k: [ToolDefinition(**t) for t in params.get(k, [])] for k in ('function_tools', 'output_tools')}
    return ModelRequestParameters(**(params | tools))


@dataclass
class State:
    """Everything needed to restore a session: the Monty dump plus the in-memory workspace, if any."""

    monty: bytes
    files: Files | None = None

    def encode_files(self) -> dict[str, dict[str, str]] | None:
        if self.files is None:
            return None
        return {
            p: {'text': c} if isinstance(c, str) else {'base64': base64.b64encode(c).decode()}
            for p, c in sorted(self.files.items())
        }

    @staticmethod
    def decode_files(encoded: dict[str, dict[str, str]] | None) -> Files | None:
        if encoded is None:
            return None
        return {p: c['text'] if 'text' in c else base64.b64decode(c['base64']) for p, c in encoded.items()}


@dataclass
class Suspended:
    """A turn stopped before a model request; `checkpoint` restarts it, with any model (a fork)."""

    checkpoint: Path


@dataclass
class StateStore:
    """`NNNN.monty` dumps with JSON sidecars (meta and workspace files); HMAC-signed when `key` is set."""

    root: Path
    key: bytes | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self) -> list[Path]:
        return sorted(self.root.glob('*.monty'))

    def save(self, state: State, meta: dict[str, Any] | None = None, name: str | None = None) -> Path:
        path = self.root / f'{name or f"{len(self.paths()):04d}"}.monty'
        files = state.encode_files()
        path.write_bytes(state.monty)
        sidecar = (meta or {}) | {'files': files, 'hmac': self._sign(state.monty, files)}
        path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2))
        return path

    def load(self, path: Path) -> State:
        blob = path.read_bytes()
        sidecar = json.loads(path.with_suffix('.json').read_text())
        if self.key and not hmac.compare_digest(sidecar.get('hmac') or '', self._sign(blob, sidecar['files']) or ''):
            raise ValueError(f'{path} failed HMAC verification')
        return State(blob, State.decode_files(sidecar['files']))

    def _sign(self, blob: bytes, files: dict | None) -> str | None:
        if not self.key:
            return None
        return hmac.new(self.key, blob + json.dumps(files, sort_keys=True).encode(), hashlib.sha256).hexdigest()


@dataclass
class Sandbox:
    """Host policy: which model answers, what the workspace is, which env vars are visible."""

    model: Model | str | None = None
    workspace: Path | None = None
    workspace_mode: WorkspaceMode = 'overlay'
    env: dict[str, str] = field(default_factory=dict)
    verbose: bool = False
    limits: ResourceLimits | None = None
    remote: str | None = None

    def host_module(self, pool: Any = None) -> ExternalModule:
        def prepare(model: str | None, messages: list[dict], params: dict):
            resolved = resolve_model(self.model or model or 'test')
            if self.verbose:
                names = [t['name'] for t in params['function_tools'] + params['output_tools']]
                log(f'model_request {resolved.model_name} messages={len(messages)} tools={names}')
            return resolved, ModelMessagesTypeAdapter.validate_python(messages), request_parameters(params)

        def model_request(model: str | None, messages: list[dict], params: dict) -> dict:
            # Monty calls sync host functions on the running loop, so the request gets a loop of its own.
            run = contextvars.copy_context().run
            with ThreadPoolExecutor(1) as executor:
                return executor.submit(run, asyncio.run, model_request_async(model, messages, params)).result()

        async def model_request_async(model: str | None, messages: list[dict], params: dict) -> dict:
            resolved, history, parameters = prepare(model, messages, params)
            response = await direct_model_request(resolved, history, model_request_parameters=parameters)
            return to_jsonable_python(response)

        async def run_python(code: str) -> dict:
            """Run `code` in a fresh child sandbox from the same pool: no files, env or host functions."""
            if self.verbose:
                log(f'run_python {len(code)} chars')
            stdout = CollectString(max_bytes=CHILD_OUTPUT_BYTES)
            checkout = {'ephemeral': True} if self.remote else {}
            with logfire.span('child sandbox', code=code) as span:
                try:
                    async with pool.checkout(script_name='child.py', limits=CHILD_LIMITS, **checkout) as child:
                        output = await child.feed_run(code, print_callback=stdout)
                    result = {'output': repr(output), 'stdout': stdout.output}
                except MontyError as error:
                    result = {'error': f'{type(error).__name__}: {error}', 'stdout': stdout.output}
                span.set_attribute('result', result)
            return result

        functions = {'model_request': model_request, 'model_request_async': model_request_async}
        if pool is not None:
            functions['run_python'] = run_python
        return ExternalModule(functions)

    def mounts(self) -> list[MountDir]:
        mounts = [MountDir(host_path=SANDBOX_DIR, virtual_path='/app', mode='read-only')]
        if self.workspace and self.workspace_mode != 'memory':
            mounts.append(MountDir(host_path=self.workspace, virtual_path=str(WORKSPACE), mode=self.workspace_mode))
        return mounts

    def seed_files(self) -> Files | None:
        if self.workspace_mode != 'memory':
            return None
        files: Files = {}
        for path in sorted(self.workspace.rglob('*')) if self.workspace else []:
            if path.is_file():
                data = path.read_bytes()
                try:
                    content: str | bytes = data.decode()
                except UnicodeDecodeError:
                    content = data
                files[str(WORKSPACE / path.relative_to(self.workspace).as_posix())] = content
        return files

    def session(self) -> 'AgentSession':
        return AgentSession(self)


class AgentSession:
    """One Monty session holding one agent and its conversation.

    In `memory` workspace mode the files live in host memory and travel with every dump, so restoring a state or
    checkpoint also restores the workspace as it was. `overlay` writes vanish at the end of each feed and on restore;
    `read-write` writes reach the host disk and are never rolled back.
    """

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self._stack = AsyncExitStack()
        self._pending: AsyncFunctionSnapshot | None = None
        self._fs: OSAccess | None = None

    async def __aenter__(self) -> Self:
        self._mounts = [self._stack.enter_context(m) for m in self.sandbox.mounts()]
        self._set_files(self.sandbox.seed_files())
        pool = await self._stack.enter_async_context(self._pool())
        self._session: AsyncMontySession = await self._stack.enter_async_context(
            pool.checkout(limits=self.sandbox.limits, external_modules={'host': self.sandbox.host_module(pool)})
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    def _pool(self) -> Any:
        if not self.sandbox.remote:
            return AsyncMonty()
        # LOGFIRE_API_KEY: a project (not personal) key with `project:execute_agent_sandboxes`.
        return logfire.agent_sandboxes(url=self.sandbox.remote, auto_resume=False)

    async def start(self, script: Path) -> None:
        await self._feed(SETUP + script.read_text(), {})

    async def send(self, prompt: str, checkpoints: StateStore | None = None, suspend_at: int | None = None) -> Any:
        """Run one turn; with `suspend_at=n`, stop before its n-th model request and return `Suspended`."""
        with logfire.span('agent turn {prompt}', prompt=prompt):
            return await self._feed(TURN, {'prompt': prompt}, checkpoints, suspend_at)

    async def dump(self) -> State:
        """Idle-session state between turns: agent, tools, history and the in-memory workspace."""
        return State(await self._session.dump(), self.files())

    async def load(self, state: State) -> None:
        await self._session.load_session(state.monty)
        self._set_files(state.files)

    async def restore(self, checkpoint: State) -> None:
        """Load a mid-turn checkpoint; `resume` then re-issues its pending model request."""
        self._set_files(checkpoint.files)
        logfire.info('restore checkpoint')
        self._pending = await self._session.load_snapshot(checkpoint.monty, mount=self._mounts, os=self._handle_os)

    async def resume(self, checkpoints: StateStore | None = None, suspend_at: int | None = None) -> Any:
        snapshot, self._pending = self._pending, None
        assert snapshot is not None, 'nothing to resume; call restore() first'
        with logfire.span('agent resume'):
            return await self._drive(snapshot, checkpoints, suspend_at)

    async def history(self) -> list[dict]:
        return await self._feed('dump_messages(history)', {})

    async def set_history(self, messages: list[dict]) -> None:
        """Replace the conversation, e.g. with a truncated or forked Pydantic AI history."""
        await self._feed(
            'from pydantic_ai_lite import load_messages\nhistory = load_messages(messages)', {'messages': messages}
        )

    def files(self) -> Files | None:
        if self._fs is None:
            return None
        return {str(f.path): f.content for f in self._fs.files if isinstance(f, MemoryFile) and not f.deleted}

    def _set_files(self, files: Files | None) -> None:
        if files is not None:
            self._fs = OSAccess([MemoryFile(p, c) for p, c in files.items()], max_sleep=None)
            self._fs.path_mkdir(WORKSPACE, parents=True, exist_ok=True)

    def _handle_os(self, *, name: str, args: tuple, **kwargs: Any) -> Any:
        if self.sandbox.verbose:
            log(f'os {name}{args}')
        env = self.sandbox.env
        if name == 'os.getenv':
            return env.get(args[0], args[1])
        if name == 'os.environ':
            return dict(env)
        if self._fs and args and isinstance(args[0], PurePosixPath) and args[0].is_relative_to(WORKSPACE):
            return self._fs(name=name, args=args, **kwargs)
        return NOT_HANDLED

    async def _feed(
        self, code: str, inputs: dict, checkpoints: StateStore | None = None, suspend_at: int | None = None
    ) -> Any:
        snapshot = await self._session.feed_start(code, inputs=inputs, mount=self._mounts, os=self._handle_os)
        return await self._drive(snapshot, checkpoints, suspend_at)

    async def _drive(self, snapshot: Any, checkpoints: StateStore | None, suspend_at: int | None = None) -> Any:
        assert suspend_at is None or checkpoints, 'suspending needs a checkpoint store'
        step = 0
        while not isinstance(snapshot, MontyComplete):
            if checkpoints and isinstance(snapshot, AsyncFunctionSnapshot) and snapshot.function_name in MODEL_REQUEST:
                step += 1
                path = checkpoints.save(State(snapshot.dump(), self.files()), {'messages': len(snapshot.args[1])})
                logfire.info('checkpoint {step}', step=step, path=str(path))
                if self.sandbox.verbose:
                    log(f'checkpoint {path}')
                if step == suspend_at:
                    logfire.info('suspended before model request {step}', step=step)
                    return Suspended(path)
            snapshot = await snapshot.resume_auto()
        return snapshot.output


async def run_once(sandbox: Sandbox, script: Path, prompt: str, checkpoints: StateStore | None = None) -> Any:
    async with sandbox.session() as session:
        await session.start(script)
        return await session.send(prompt, checkpoints)


def run(script: Path, *, prompt: str, **sandbox: Any) -> Any:
    return asyncio.run(run_once(Sandbox(**sandbox), script, prompt))


async def cli(args: argparse.Namespace) -> Any:
    sandbox = Sandbox(
        model=args.model,
        workspace=args.workspace,
        workspace_mode=args.workspace_mode,
        env={k: os.environ[k] for k in args.env if k in os.environ},
        verbose=args.verbose,
        limits=ResourceLimits(max_memory=100_000_000, max_feed_duration_secs=30),
        remote=args.remote,
    )
    key = os.environ.get('MONTY_AGENT_KEY', '').encode() or None
    checkpoints = StateStore(args.checkpoints, key) if args.checkpoints else None
    async with sandbox.session() as session:
        if args.resume:
            await session.restore(StateStore(args.resume.parent, key).load(args.resume))
            output = await session.resume(checkpoints, args.suspend_at)
        else:
            if args.state and args.state.exists():
                await session.load(StateStore(args.state.parent, key).load(args.state))
            else:
                await session.start(args.script)
            output = await session.send(args.prompt, checkpoints, args.suspend_at)
        if isinstance(output, Suspended):
            log(f'suspended; continue or fork with --resume {output.checkpoint}')
            return None
        if args.state:
            StateStore(args.state.parent, key).save(await session.dump(), name=args.state.stem)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('script', type=Path, nargs='?')
    parser.add_argument('prompt', nargs='?')
    parser.add_argument('--model', help="e.g. 'test', 'openai:gpt-5.2', 'anthropic:claude-sonnet-5-5'")
    parser.add_argument('--workspace', type=Path, help='host directory exposed at /workspace')
    parser.add_argument('--workspace-mode', choices=['overlay', 'read-write', 'memory'], default='overlay')
    parser.add_argument('--env', action='append', default=[], help='host env var visible to the sandbox')
    parser.add_argument('--state', type=Path, help='idle session file (.monty): loaded if present, saved after')
    parser.add_argument('--checkpoints', type=Path, help='directory for a checkpoint before every model request')
    parser.add_argument('--resume', type=Path, help='checkpoint to restore and finish (fork / roll back)')
    parser.add_argument(
        '--suspend-at', type=int, metavar='N', help='stop before the N-th model request (needs --checkpoints)'
    )
    parser.add_argument(
        '--remote', metavar='URL', help='run in Logfire Agent Sandboxes, e.g. https://logfire-eu.pydantic.info/'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if not args.resume and not (args.script and args.prompt):
        parser.error('script and prompt are required unless --resume is given')
    if args.suspend_at and not args.checkpoints:
        parser.error('--suspend-at needs --checkpoints')

    configure_logfire()
    with logfire.span('monty-agent {script}', script=str(args.script or args.resume)):
        output = asyncio.run(cli(args))
    if output is not None:
        print(getattr(output, 'attributes', output))


if __name__ == '__main__':
    main()
