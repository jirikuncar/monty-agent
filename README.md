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
| Data-level fork          | `history()` / `set_history(messages)`                    | Pydantic AI JSON messages only              |

A fork is a restore into a fresh session with a different model (or prompt); sessions never share state.

Workspace modes decide what a restore does to files:

| `--workspace-mode` | Writes go to                      | On restore                                                  |
| ------------------ | --------------------------------- | ----------------------------------------------------------- |
| `overlay`          | Monty memory, discarded per feed  | earlier writes are gone, so tools may see missing files     |
| `read-write`       | host disk                         | nothing is undone; tool side effects run again              |
| `memory`           | host `OSAccess`, saved with state | files come back exactly as they were at the checkpoint      |

Use `memory` for anything you want to fork or roll back. Real external side effects (HTTP, databases) behave like `read-write`: make those tools idempotent.

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
