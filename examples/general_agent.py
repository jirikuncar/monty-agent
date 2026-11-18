"""A general-purpose agent: plans, delegates to specialist sub-agents and solves problems in code mode.

Code mode: instead of a chain of tool calls, the model writes one Python program, runs it in a fresh child sandbox
with `run_code` and gets back its last expression. Workspace files it names are passed in as the `files` global.

Specialists are agents of their own, in this same sandbox and session: each has its own instructions, tools and
message history, and a checkpoint taken during a delegation restores the whole stack of agents.
"""

from pathlib import Path

from pydantic_ai_lite import Agent

from host import ask_user, run_python

WORKSPACE = Path('/workspace')


async def run_code(code: str, files: list[str] | None = None) -> dict:
    inputs = {'files': {name: (WORKSPACE / name).read_text() for name in files or []}}
    return await run_python(code, inputs)


def list_files() -> list[str]:
    if not WORKSPACE.exists():
        return []
    prefix = len(str(WORKSPACE)) + 1
    return sorted([str(p)[prefix:] for p in WORKSPACE.rglob('*') if p.is_file()])


def read_file(name: str) -> str:
    return (WORKSPACE / name).read_text()


def write_file(name: str, text: str) -> str:
    path = WORKSPACE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


CODE_MODE = (
    'Solve anything computational in code mode: write one complete Python program and run it with `run_code`. '
    'Its last expression is returned and `print` output is captured. Workspace files named in `files` are '
    'available as the `files` dict (name -> text). Only the Monty subset of the standard library exists: '
    'no file, network or third-party access. On an error, fix the program and run it again. '
)
RUN_CODE = 'Run a Python program in a fresh, isolated sandbox; `files` lists workspace files to pass in.'
LIST_FILES = 'List the workspace files.'
READ_FILE = 'Read a workspace file.'
WRITE_FILE = 'Write a workspace file; returns its path.'

coder = Agent(
    instructions=CODE_MODE + 'Answer with the result and the final program.',
    output_type=str,
)
coder.tool_plain(run_code, name='run_code', description=RUN_CODE)

analyst = Agent(
    instructions=(
        'Analyse the workspace files for the task: find the relevant ones, read them, and compute anything '
        'quantitative in code mode. ' + CODE_MODE + 'Answer with findings and the evidence for each.'
    ),
    output_type=str,
)
analyst.tool_plain(list_files, name='list_files', description=LIST_FILES)
analyst.tool_plain(read_file, name='read_file', description=READ_FILE)
analyst.tool_plain(run_code, name='run_code', description=RUN_CODE)

writer = Agent(
    instructions=(
        'Write or revise the requested text: clear, concise, in the format asked for. Save it with `write_file` '
        'when a file name is given. Answer with the path, if saved, and the text.'
    ),
    output_type=str,
)
writer.tool_plain(read_file, name='read_file', description=READ_FILE)
writer.tool_plain(write_file, name='write_file', description=WRITE_FILE)

critic = Agent(
    instructions=(
        'Review the work against the task. List concrete problems (wrong results, missing requirements, '
        'unclear text), most important first, or say it is ready. Do not rewrite it.'
    ),
    output_type=str,
)

SPECIALISTS = {
    'coder': (coder, 'computation, algorithms and data transformation in code mode'),
    'analyst': (analyst, 'questions about the workspace files, with code-mode analysis'),
    'writer': (writer, 'drafting and revising text, saved to the workspace'),
    'critic': (critic, 'reviewing a result or draft against the task'),
}

agent = Agent(
    instructions=(
        'You are a general-purpose assistant. Plan the task, then solve it with the fewest steps: '
        'do simple things yourself, delegate a self-contained subtask to a specialist with `delegate`, and run '
        'code for anything computational. ' + CODE_MODE + 'A specialist sees only the task you give it, '
        'so include every fact it needs. Delegate one task at a time. For non-trivial results, have the critic '
        'review them and fix what it finds. Ask the user with `ask_user` only when the task cannot be done '
        'without their answer, one question at a time. Answer with the result and, briefly, how you got it.'
    ),
    output_type=str,
)


@agent.tool_plain(
    name='delegate',
    description='Give a self-contained task to a specialist and return its answer. Specialists: '
    + '; '.join(f'{name}: {about}' for name, (_, about) in SPECIALISTS.items()),
)
async def delegate(specialist: str, task: str) -> str:
    if specialist not in SPECIALISTS:
        raise ValueError(f'unknown specialist {specialist!r}; choose one of {sorted(SPECIALISTS)}')
    result = await SPECIALISTS[specialist][0].run(task, max_steps=10)
    return result.output


@agent.tool_plain(name='ask_user', description='Ask the user one question and wait for the answer.')
async def ask_user_tool(question: str) -> str:
    return await ask_user(question)


agent.tool_plain(run_code, name='run_code', description=RUN_CODE)
agent.tool_plain(list_files, name='list_files', description=LIST_FILES)
agent.tool_plain(read_file, name='read_file', description=READ_FILE)
agent.tool_plain(write_file, name='write_file', description=WRITE_FILE)
