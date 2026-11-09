"""Suspend an agent mid-turn, restart it from a step in a fresh session, and fork another step with a different model.

    uv run examples/suspend_fork.py
    uv run examples/suspend_fork.py --model gateway/anthropic:claude-haiku-5-5 --fork-model ollama:qwen3.5:latest
    uv run --extra remote examples/suspend_fork.py --remote https://logfire-eu.pydantic.info/ ...

Every step is a checkpoint taken before a model request. The workspace lives in memory and travels with each
checkpoint, so a restart or fork sees the files exactly as they were at that step.
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logfire

from host import Sandbox, StateStore, Suspended, configure_logfire

AGENT = Path(__file__).parent / 'notes_agent.py'
PROMPT = 'Read my notes, then write summary.md with a one-line summary.'


async def main(args: argparse.Namespace) -> None:
    root = Path(tempfile.mkdtemp(prefix='suspend-fork-'))
    workspace = root / 'notes'
    workspace.mkdir()
    (workspace / 'todo.md').write_text('Buy milk. Meeting with Samuel on Friday about Monty.\n')

    def sandbox(model: str) -> Sandbox:
        return Sandbox(model=model, workspace=workspace, workspace_mode='memory', remote=args.remote)

    # 1. Run until just before the `suspend_at`-th model request, then drop the session.
    steps = StateStore(root / 'run')
    with logfire.span('suspend before model request {step}', step=args.suspend_at):
        async with sandbox(args.model).session() as session:
            await session.start(AGENT)
            suspended = await session.send(PROMPT, steps, suspend_at=args.suspend_at)
    if not isinstance(suspended, Suspended):
        sys.exit(f'finished in fewer than {args.suspend_at} model requests: {suspended}')
    print(f'suspended before model request {args.suspend_at}: {suspended.checkpoint}')
    for path in steps.paths():
        print(f'  step {path.stem}: files={sorted(steps.load(path).files or {})}')

    # 2. Restart from the suspended step in a fresh session, as another process would.
    with logfire.span('restart from step {step}', step=suspended.checkpoint.stem):
        async with sandbox(args.model).session() as session:
            await session.restore(steps.load(suspended.checkpoint))
            output = await session.resume(StateStore(root / 'restart'))
            print(f'restart  -> {getattr(output, "attributes", output)}\n  files: {sorted(session.files() or {})}')

    # 3. Fork an earlier step with another model; the original run is untouched.
    fork_from = steps.paths()[args.fork_step - 1]
    model = args.fork_model or args.model
    with logfire.span('fork step {step} with {model}', step=fork_from.stem, model=model):
        async with sandbox(model).session() as session:
            await session.restore(steps.load(fork_from))
            output = await session.resume(StateStore(root / f'fork-{fork_from.stem}'))
            files = sorted(session.files() or {})
            print(f'fork {fork_from.stem} -> {getattr(output, "attributes", output)}\n  files: {files}')
    print(f'checkpoints: {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='test')
    parser.add_argument('--fork-model', help='model for the fork (default: --model)')
    parser.add_argument('--suspend-at', type=int, default=2, help='suspend before this model request')
    parser.add_argument('--fork-step', type=int, default=1, help='step to fork from')
    parser.add_argument('--remote', metavar='URL', help='run in Logfire Agent Sandboxes')
    configure_logfire()
    with logfire.span('suspend/fork walkthrough'):
        asyncio.run(main(parser.parse_args()))
