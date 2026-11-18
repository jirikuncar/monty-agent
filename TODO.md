# TODO

## Volumes for agent data

Use Agent Sandboxes volumes for data the agent keeps outside the conversation (notes, datasets, results shared
between conversations); checkpoints stay in Logfire.

- `request_volume(name, mode, reason)` suspends as a question node; approval attaches the volume to the
  conversation's profile and the resume (always a fresh session) sees it at `/volumes/<name>`.
- Forks and re-answers inherit the profile; volume contents are not rolled back.
- Blocked: volumes and profiles are managed only through the Logfire UI's API, which sends OAuth bearers to the
  browser login. Needs OAuth-reachable endpoints or MCP tools, under a scope such as
  `project:manage_agent_sandboxes`.
- Works today without that: profiles made in the UI, passed with `checkout(profile=...)`.

## TUI

- Tests for `oauth.login`/`refresh` (mocked endpoints) and the TUI token loop; `tests/test_tui.py` covers the
  conversation view, polling, forks, retry and archiving against a fake Logfire.
- `export_with` patches `logfire._internal.config.OTLPExporterHttpSession`; ask for a public way to rotate the
  export token in the Logfire SDK.
- `logfire.agent_sandboxes()` falls back to the `api_key` `configure()` read at startup, so a refreshed token
  only reaches pools passed `api_key=` (`host.sandbox_key`). Ask for it to read the current key, like the exporter.
- Code runs show once a turn ends (they are read from its trace); stream them live through `on_record`-style hooks.
- README: compare the `span` and `session` files modes.
- `McpQuery` keeps its MCP session ID until a token refresh: on a 404/400 for an expired session, initialize again.
  Two concurrent first queries both initialize. SSE replies with several `data:` events join into invalid JSON.
- `keep_fresh` stops on the first failed refresh; retry transient errors with backoff. A rotated refresh token can
  be lost when "Refresh token" or "Log in again" cancels an in-flight refresh (exclusive `token` group).
- A pool reads `sandbox_key()` once: a turn that outlives the token gets 401s on later checkouts.
- `oauth.login`: a 5xx HTML body while polling aborts the login; an unknown backend raises a bare `KeyError`
  (`GATEWAYS`); every login registers a new client.
- The staging query path is the Logfire MCP `query_run` (14-day window); switch to `/v2/query` once it accepts
  OAuth tokens with `project:read_otlp`.
