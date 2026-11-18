from dataclasses import dataclass
from pathlib import Path

from pydantic_ai_lite import Agent

from host import ask_user


@dataclass
class Invitation:
    path: str
    text: str


agent = Agent(
    instructions=(
        'Draft a short meeting invitation and save it under /workspace with `write_note`. '
        'Never guess the attendees, date, time or topic: if the request leaves any of them out, ask the user with '
        '`ask_user`, one question at a time. Return the saved path and the text.'
    ),
    output_type=Invitation,
)


@agent.tool_plain(name='ask_user', description='Ask the user one question and wait for the answer.')
async def ask_user_tool(question: str) -> str:
    return await ask_user(question)


@agent.tool_plain(description='Write a note to /workspace/<name>; returns the path.')
def write_note(name: str, text: str) -> str:
    path = Path('/workspace') / Path(name).name
    path.write_text(text)
    return str(path)
