# monty-agent

A minimal Pydantic AI subset that runs inside the [Monty](https://github.com/pydantic/monty) sandbox.
The agent loop, tools, structured output and message types run in the sandbox; every external call is answered by the host:

| Sandbox code                         | Host side                                                          |
| ------------------------------------ | ------------------------------------------------------------------ |
| `agent.run` / `run_sync` -> LLM      | `host.model_request[_async]` external module -> `pydantic_ai.direct` |
| `open`, `pathlib`, `os.listdir`, ... | `/app` read-only mount (the library), `/workspace` (see modes)     |
| `os.getenv`, `os.environ`            | `os` handler, allowlisted vars only                                |

API keys never enter the sandbox: the host owns the model client.

```sh
uv run host.py examples/notes_agent.py "summarize my notes" --workspace ./notes --env USER -v
uv run host.py examples/notes_agent.py "add a note" --model anthropic:claude-sonnet-5-5 --workspace ./notes --workspace-mode read-write
uv run pytest
```

`--model test` (default) uses `TestModel`, which calls every tool once and then the output tool.

## Logfire Agent Sandboxes and AI Gateway

`--remote URL` runs the same session in managed Logfire Agent Sandboxes instead of local Monty; the host still answers
every model request, file and env call, so `/app`, `/workspace` and the model client stay on your machine.

```sh
uv sync --extra remote
export LOGFIRE_API_KEY=...              # project key with `project:execute_agent_sandboxes`
export PYDANTIC_AI_GATEWAY_API_KEY=...  # project key with AI Gateway access (can be the same key)
uv run host.py examples/notes_agent.py "summarize my notes" --workspace ./notes \
  --model gateway/anthropic:claude-haiku-5-5 --remote https://logfire-eu.pydantic.info/
```

Create the key in the Logfire UI (Agent Sandboxes > Setup enables sandboxes for the project). It must be a project key:
the sandbox relay rejects personal keys and OAuth tokens. The Gateway URL is inferred from the key's region.
Local models work too, e.g. `--model ollama:qwen3.5:latest` with `OLLAMA_BASE_URL=http://localhost:11434/v1`.

## Child sandboxes

An agent can run code it writes in another Monty sandbox: `host.run_python(code)` checks out a fresh session from the
same pool (local `AsyncMonty` or Agent Sandboxes with `--remote`) and returns `{'output': repr(result), 'stdout': ...}`
or `{'error': ..., 'stdout': ...}`. The child gets no mounts, no env, no host functions and tighter limits
(`CHILD_LIMITS`: 50 MB, 10 s), so model-written code reaches nothing the parent can; errors go back to the model.

```sh
uv run host.py examples/code_agent.py "What is the 30th Fibonacci number?" \
  --model gateway/anthropic:claude-haiku-5-5 --remote https://logfire-eu.pydantic.info/ -v
```

With tracing on, the child session nests under the host's `child sandbox` span, in the same trace as the parent session
and the Gateway calls (staging run, `os call` spans elided):

```text
monty-agent examples/code_agent.py              monty-agent-host
├─ session main.py                              monty-worker (parent)
│  └─ run code
│     ├─ call host.model_request_async
│     ├─ call host.run_python
│     └─ call host.model_request_async
└─ agent turn What is the 30th Fibonacci ...    monty-agent-host
   ├─ chat claude-haiku-5-5 ─ anthropic.messages.create ─ chat claude-haiku-5-5   PAIG
   ├─ child sandbox                             monty-agent-host (attributes: code, result)
   │  └─ session child.py                       monty-worker (child)
   │     └─ run code ─ print stdout
   └─ chat claude-haiku-5-5 ─ anthropic.messages.create ─ chat claude-haiku-5-5   PAIG
```

## Subset

- `Agent(model, instructions=, output_type=, retries=, output_retries=)`; `output_type` is `str`, a scalar or an annotated class/dataclass (output tool `final_result`)
- `@agent.tool` (with `RunContext`), `@agent.tool_plain`; tools may be `async def`
- `await agent.run(...)` runs tool calls concurrently; `agent.run_sync(...)` runs them in order
- `ModelRequest`, `ModelResponse`, `UserPromptPart`, `TextPart`, `ToolCallPart`, `ToolReturnPart`, `RetryFeedbackPart`; anything else round-trips as `RawPart`
- `dump_messages` / `load_messages` match `ModelMessagesTypeAdapter` JSON, so histories move between this and real Pydantic AI
- Tool errors, unknown tools and invalid output go back as `ToolReturnPart(outcome='retried')` (Pydantic AI 2.x replaced `RetryPromptPart`)

## Store, fork, roll back

`AgentSession` keeps one agent and its `history` in one Monty session. State is a Monty dump plus, in `memory` mode, the workspace files, stored by `StateStore` (HMAC-signed with `MONTY_AGENT_KEY`).

| What                     | API / CLI                                                | Contains                                    |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------- |
| Between turns            | `session.dump()` / `load()`, `--state chat.monty`        | agent, tools, history, retry state          |
| Before each model request| `send(prompt, checkpoints)`, `--checkpoints DIR`         | the paused call stack too                   |
| Resume, fork, roll back  | `restore(state)` + `resume()`, `--resume DIR/0001.monty` | re-issues the pending model request         |
| Suspend at step N        | `send(..., suspend_at=N)`, `--suspend-at N`              | returns `Suspended(checkpoint)` before request N |
| Data-level fork          | `history()` / `set_history(messages)`                    | Pydantic AI JSON messages only              |

A fork is a restore into a fresh session with a different model (or prompt); sessions never share state.

Workspace modes decide what a restore does to files:

| `--workspace-mode` | Writes go to                      | On restore                                                  |
| ------------------ | --------------------------------- | ----------------------------------------------------------- |
| `overlay`          | Monty memory, discarded per feed  | earlier writes are gone, so tools may see missing files     |
| `read-write`       | host disk                         | nothing is undone; tool side effects run again              |
| `memory`           | host `OSAccess`, saved with state | files come back exactly as they were at the checkpoint      |

Use `memory` for anything you want to fork or roll back. Real external side effects (HTTP, databases) behave like `read-write`: make those tools idempotent.

`examples/suspend_fork.py` walks through it: suspend before the second model request, restart that step in a fresh
session, and fork the first step with another model. It runs offline with `TestModel` by default:

```sh
uv run examples/suspend_fork.py
uv run examples/suspend_fork.py --model gateway/anthropic:claude-haiku-5-5 --fork-model ollama:qwen3.5:latest \
  --remote https://logfire-eu.pydantic.info/
```

### Traces

With `LOGFIRE_TOKEN` (a project write token) set, the host exports its spans and Pydantic AI model requests, and both
sides of every model call join the same trace: the remote sandbox session gets the host's `traceparent` when it is
checked out, and the Gateway provider sends `traceparent` with each request. One walkthrough run on staging
(`--model gateway/anthropic:claude-haiku-5-5 --fork-model ollama:qwen3.5:latest --remote ...`), with `os call` spans
and repeated steps elided:

```text
suspend/fork walkthrough                                   monty-agent-host
├─ suspend before model request 2                          monty-agent-host
│  ├─ session main.py                                      monty-worker
│  │  └─ run code
│  │     ├─ call host.model_request_async ─ dump           (step 1)
│  │     └─ call host.model_request_async ─ dump           (step 2, suspended)
│  └─ agent turn Read my notes, then write summary.md ...  monty-agent-host
│     ├─ checkpoint 1
│     ├─ chat claude-haiku-5-5 ─ anthropic.messages.create
│     │  └─ chat claude-haiku-5-5                          PAIG (Gateway)
│     ├─ checkpoint 2
│     └─ suspended before model request 2
├─ restart from step 0001                                  monty-agent-host
│  ├─ session main.py                                      monty-worker
│  │  ├─ load ─ call host.model_request_async ─ dump
│  │  ├─ call host.model_request_async ─ dump              (x2)
│  │  └─ complete
│  ├─ restore checkpoint
│  └─ agent resume
│     ├─ checkpoint 1 · chat claude-haiku-5-5 ─ PAIG
│     ├─ checkpoint 2 · chat claude-haiku-5-5 ─ PAIG
│     └─ checkpoint 3 · chat claude-haiku-5-5 ─ PAIG
└─ fork step 0000 with ollama:qwen3.5:latest               monty-agent-host
   ├─ session main.py                                      monty-worker
   │  ├─ load ─ call host.model_request_async ─ dump
   │  ├─ call host.model_request_async ─ dump              (x4)
   │  └─ complete
   ├─ restore checkpoint
   └─ agent resume
      └─ checkpoint 1..5 · chat qwen3.5:latest             (local, no Gateway span)
```

Steps in a restart or fork are numbered from the restored checkpoint, and each branch writes its own checkpoint
directory (`run/`, `restart/`, `fork-0000/`), so the original run is never overwritten. Find a run with
`SELECT trace_id FROM records WHERE span_name = 'suspend/fork walkthrough'`.

The same from the CLI, one process per step:

```sh
uv run host.py examples/notes_agent.py "summarize my notes" --workspace ./notes --workspace-mode memory \
  --checkpoints run --suspend-at 2                                        # suspend
uv run host.py --resume run/0001.monty --workspace-mode memory             # restart from step 2
uv run host.py --resume run/0000.monty --workspace-mode memory --model ollama:qwen3.5:latest  # fork step 1
```

Other caveats:

- A dump holds compiled code: editing `pydantic_ai_lite.py` or the agent script does not affect a restored session, and a Monty release that bumps its dump format refuses older dumps.
- Overlay writes do not survive between turns either, since each turn is its own feed.
- Only files are captured in `memory` mode; empty directories are not.
- Checkpoints are taken only at model requests, which are awaited eagerly. A dump taken while host coroutines are pending (e.g. inside `asyncio.gather` over host calls) cannot be driven by `resume_auto()` after restore.
- Only restore dumps you produced; Monty does not authenticate them, hence the HMAC.

## Monty constraints that shape the code

- Functions have no `__name__` or `__doc__`: tool names come from `repr(func)`, descriptions are `description=`.
- No `typing.get_origin`, `dataclasses.asdict`/`fields` or class inheritance (so no `ModelRetry`): schemas read `__origin__`/`__args__`, parts serialize via `to_dict`, any tool exception is a retry.
- Parameter defaults are not introspectable: only `X | None` parameters and fields are optional.
- Output validation is shallow: scalars, `list`, `dict` and unions are checked; nested classes are not built.
