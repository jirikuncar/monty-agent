import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai_lite import Agent, RunContext


@dataclass
class Answer:
    summary: str
    files: list[str]


agent = Agent(
    instructions='You manage text notes in the workspace. Use the tools, then answer briefly.',
    output_type=Answer,
)
root = Path('/workspace')
deps = os.getenv('USER', 'anonymous')


@agent.tool_plain(description='List note file names.')
def list_notes() -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_file()])


@agent.tool_plain(description='Read a note by file name.')
def read_note(name: str) -> str:
    return (root / name).read_text()


@agent.tool(description='Write a note, signed with the current user.')
async def write_note(ctx: RunContext, name: str, content: str) -> str:
    (root / name).write_text(f'{content}\n-- {ctx.deps}\n')
    return f'wrote {name}'
