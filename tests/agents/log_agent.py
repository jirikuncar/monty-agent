from pathlib import Path

from pydantic_ai_lite import Agent

agent = Agent()
log = Path('/workspace/log.txt')


@agent.tool_plain(description='Append a line to the log; returns the line count.')
def append(line: str) -> int:
    text = log.read_text() if log.exists() else ''
    log.write_text(text + line + '\n')
    return len(text.splitlines()) + 1
