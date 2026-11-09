from dataclasses import dataclass

from pydantic_ai_lite import Agent

from host import run_python


@dataclass
class Answer:
    answer: str
    code: str


agent = Agent(
    instructions=(
        'Answer by computing, not guessing. Write Python and run it with `run_python`; '
        'its last expression is returned and `print` output is captured. '
        'Only the standard library subset of Monty is available: no files, network or third-party packages. '
        'If it reports an error, fix the code and run it again. Return the answer and the final code.'
    ),
    output_type=Answer,
)


@agent.tool_plain(name='run_python', description='Run Python code in a fresh, isolated sandbox and return its result.')
async def run_python_tool(code: str) -> dict:
    return await run_python(code)
