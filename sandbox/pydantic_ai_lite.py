"""Minimal Pydantic AI subset that runs inside the Monty sandbox.

The only way out is the host: `host.model_request` for the LLM, mounts for files, the os handler for env.
Message dicts follow the Pydantic AI wire format so the host can validate them with `ModelMessagesTypeAdapter`.
"""

import asyncio
import json
import typing
from dataclasses import dataclass, field

from host import model_request, model_request_async

FINAL_RESULT_TOOL = 'final_result'


@dataclass
class UserPromptPart:
    content: object
    raw: dict = field(default_factory=dict)
    part_kind: str = 'user-prompt'

    def to_dict(self):
        return self.raw | {'part_kind': self.part_kind, 'content': self.content}


@dataclass
class ToolReturnPart:
    tool_name: str
    content: object
    tool_call_id: str
    outcome: str = 'success'
    retry: dict | None = None
    raw: dict = field(default_factory=dict)
    part_kind: str = 'tool-return'

    def to_dict(self):
        return self.raw | {
            'part_kind': self.part_kind,
            'tool_name': self.tool_name,
            'content': self.content,
            'tool_call_id': self.tool_call_id,
            'outcome': self.outcome,
            'retry': self.retry,
        }


@dataclass
class RetryFeedbackPart:
    content: str
    retry: dict
    part_kind: str = 'retry-feedback'

    def to_dict(self):
        return {'part_kind': self.part_kind, 'content': self.content, 'retry': self.retry}


@dataclass
class TextPart:
    content: str
    raw: dict = field(default_factory=dict)
    part_kind: str = 'text'

    def to_dict(self):
        return self.raw | {'part_kind': self.part_kind, 'content': self.content}


@dataclass
class ToolCallPart:
    tool_name: str
    args: object
    tool_call_id: str
    raw: dict = field(default_factory=dict)
    part_kind: str = 'tool-call'

    def args_as_dict(self):
        if not self.args:
            return {}
        return json.loads(self.args) if isinstance(self.args, str) else self.args

    def to_dict(self):
        return self.raw | {
            'part_kind': self.part_kind,
            'tool_name': self.tool_name,
            'args': self.args,
            'tool_call_id': self.tool_call_id,
        }


@dataclass
class RawPart:
    """A part this subset does not model (thinking, files, builtin tools, ...), kept for the round trip."""

    raw: dict
    part_kind: str = ''

    def to_dict(self):
        return self.raw


@dataclass
class ModelRequest:
    parts: list
    instructions: str | None = None
    raw: dict = field(default_factory=dict)
    kind: str = 'request'

    def to_dict(self):
        parts = [p.to_dict() for p in self.parts]
        return self.raw | {'kind': self.kind, 'parts': parts, 'instructions': self.instructions}


@dataclass
class ModelResponse:
    parts: list
    raw: dict = field(default_factory=dict)
    kind: str = 'response'

    def usage(self):
        return self.raw.get('usage') or {}

    def to_dict(self):
        return self.raw | {'kind': self.kind, 'parts': [p.to_dict() for p in self.parts]}


def _load_part(d):
    kind = d['part_kind']
    if kind == 'text':
        return TextPart(d['content'], d)
    if kind == 'tool-call':
        return ToolCallPart(d['tool_name'], d['args'], d['tool_call_id'], d)
    if kind == 'user-prompt':
        return UserPromptPart(d['content'], d)
    if kind == 'tool-return':
        outcome = d.get('outcome') or 'success'
        return ToolReturnPart(d['tool_name'], d['content'], d['tool_call_id'], outcome, d.get('retry'), d)
    return RawPart(d, kind)


def load_messages(dicts):
    """Inverse of `dump_messages`; accepts Pydantic AI `ModelMessagesTypeAdapter.dump_python(mode='json')` output."""
    messages = []
    for d in dicts:
        parts = [_load_part(p) for p in d['parts']]
        if d['kind'] == 'request':
            messages.append(ModelRequest(parts, d.get('instructions'), d))
        else:
            messages.append(ModelResponse(parts, d))
    return messages


def dump_messages(messages):
    return [m.to_dict() for m in messages]


@dataclass
class RunContext:
    deps: object
    messages: list
    tool_name: str
    retry: int
    run_step: int


_JSON_TYPES = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'NoneType': 'null'}
_SCALARS = ('str', 'int', 'float', 'bool', 'NoneType')


def _type_name(t):
    return getattr(t, '__name__', None) or repr(t)


def _is_union(origin):
    return _type_name(origin) in ('UnionType', 'Union')


def _json_schema(t):
    origin = getattr(t, '__origin__', None)
    args = getattr(t, '__args__', None) or ()
    if origin is None:
        name = _type_name(t)
        if name in _JSON_TYPES:
            return {'type': _JSON_TYPES[name]}
        if name == 'list':
            return {'type': 'array'}
        if name == 'dict':
            return {'type': 'object'}
        try:
            hints = typing.get_type_hints(t)
        except Exception:
            return {}
        return _object_schema(hints) if hints else {}
    if _is_union(origin):
        return {'anyOf': [_json_schema(a) for a in args]}
    name = _type_name(origin)
    if name in ('list', 'tuple', 'set'):
        return {'type': 'array', 'items': _json_schema(args[0])} if args else {'type': 'array'}
    if name == 'dict':
        return {'type': 'object', 'additionalProperties': _json_schema(args[1])} if args else {'type': 'object'}
    return {}


def _object_schema(hints):
    properties = {p: _json_schema(h) for p, h in hints.items()}
    return {
        'type': 'object',
        'properties': properties,
        'required': [p for p, s in properties.items() if {'type': 'null'} not in s.get('anyOf', [])],
        'additionalProperties': False,
    }


def _validate(value, hint, path='value'):
    """Return a list of error strings; types other than scalars, list, dict and unions pass unchecked."""
    origin = getattr(hint, '__origin__', None)
    args = getattr(hint, '__args__', None) or ()
    if origin is not None and _is_union(origin):
        for a in args:
            if not _validate(value, a, path):
                return []
        return [f'{path}: expected {hint!r}, got {value!r}']
    name = _type_name(origin or hint)
    if name == 'NoneType':
        return [] if value is None else [f'{path}: expected None, got {value!r}']
    if name == 'float':
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return [] if ok else [f'{path}: expected number, got {value!r}']
    if name in ('int', 'str', 'bool'):
        ok = isinstance(value, origin or hint) and (name == 'bool' or not isinstance(value, bool))
        return [] if ok else [f'{path}: expected {name}, got {value!r}']
    if name == 'list':
        if not isinstance(value, list):
            return [f'{path}: expected list, got {value!r}']
        errors = []
        for i, item in enumerate(value):
            errors.extend(_validate(item, args[0], f'{path}[{i}]') if args else [])
        return errors
    if name == 'dict':
        if not isinstance(value, dict):
            return [f'{path}: expected dict, got {value!r}']
        errors = []
        for k, v in value.items():
            errors.extend(_validate(v, args[1], f'{path}[{k!r}]') if args else [])
        return errors
    return []


def _build_output(output_type, args):
    """Return `(output, errors)` for an output tool call."""
    name = _type_name(output_type)
    if name in _SCALARS or getattr(output_type, '__origin__', None) is not None:
        value = args.get('response')
        return value, _validate(value, output_type, 'response')
    hints = typing.get_type_hints(output_type)
    errors = [f'{k}: unexpected field' for k in args if k not in hints]
    for k, hint in hints.items():
        if k in args:
            errors.extend(_validate(args[k], hint, k))
        elif {'type': 'null'} not in _json_schema(hint).get('anyOf', []):
            errors.append(f'{k}: field required')
    return (None, errors) if errors else (output_type(**args), [])


@dataclass
class Tool:
    function: object
    name: str
    description: str
    takes_ctx: bool
    max_retries: int

    def definition(self):
        hints = typing.get_type_hints(self.function)
        params = {p: h for p, h in hints.items() if p != 'return' and _type_name(h) != 'RunContext'}
        schema = _object_schema(params)
        return {'name': self.name, 'description': self.description, 'parameters_json_schema': schema}


def _function_name(func):
    # Monty functions have no `__name__`; their repr is `<function name at 0x..>`.
    return repr(func).split(' ')[1].split('.')[-1]


@dataclass
class AgentRunResult:
    output: object
    messages: list
    new_start: int
    usage_totals: dict

    def all_messages(self):
        return list(self.messages)

    def new_messages(self):
        return self.messages[self.new_start :]

    def all_messages_json(self):
        return dump_messages(self.messages)

    def usage(self):
        return dict(self.usage_totals)


@dataclass
class _RunState:
    deps: object
    messages: list
    new_start: int
    usage: dict = field(default_factory=lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0})
    retries: dict = field(default_factory=dict)
    output_retries: int = 0
    step: int = 0
    done: bool = False
    output: object = None


class Agent:
    def __init__(self, model=None, *, instructions=None, output_type=str, retries=1, output_retries=None):
        self.model = model
        self.instructions = instructions
        self.output_type = output_type
        self.retries = retries
        self.output_retries = retries if output_retries is None else output_retries
        self.tools = {}

    def tool(self, func=None, *, name=None, description='', retries=None):
        """Register a tool whose first argument is a `RunContext`."""
        return self._register(func, True, name, description, retries)

    def tool_plain(self, func=None, *, name=None, description='', retries=None):
        return self._register(func, False, name, description, retries)

    def _register(self, func, takes_ctx, name, description, retries):
        if func is None:
            return lambda f: self._register(f, takes_ctx, name, description, retries)
        tool_name = name or _function_name(func)
        max_retries = self.retries if retries is None else retries
        self.tools[tool_name] = Tool(func, tool_name, description, takes_ctx, max_retries)
        return func

    def run_sync(self, user_prompt, *, deps=None, message_history=None, max_steps=20):
        state = self._start(user_prompt, deps, message_history)
        params = self._params()
        while self._next_step(state, max_steps):
            response = model_request(self.model, dump_messages(state.messages), params)
            calls = self._handle_response(state, response)
            parts = [self._call_tool(state, call) for call in calls]
            self._finish_step(state, parts)
        return self._result(state)

    async def run(self, user_prompt, *, deps=None, message_history=None, max_steps=20):
        """Like `run_sync`, but awaits the host and runs tool calls concurrently; tools may be `async def`."""
        state = self._start(user_prompt, deps, message_history)
        params = self._params()
        while self._next_step(state, max_steps):
            response = await model_request_async(self.model, dump_messages(state.messages), params)
            calls = self._handle_response(state, response)
            parts = await asyncio.gather(*[self._call_tool_async(state, call) for call in calls])
            self._finish_step(state, parts)
        return self._result(state)

    def _start(self, user_prompt, deps, message_history):
        messages = list(message_history or [])
        state = _RunState(deps, messages, len(messages))
        messages.append(ModelRequest([UserPromptPart(user_prompt)], self.instructions))
        return state

    def _next_step(self, state, max_steps):
        if state.done:
            return False
        if state.step >= max_steps:
            raise RuntimeError(f'Exceeded max_steps={max_steps}')
        state.step += 1
        return True

    def _params(self):
        params = {
            'function_tools': [t.definition() for t in self.tools.values()],
            'output_tools': [],
            'allow_text_output': True,
        }
        if self.output_type is not str:
            hints = {'response': self.output_type} if _type_name(self.output_type) in _SCALARS else None
            schema = _object_schema(hints) if hints else _json_schema(self.output_type)
            output_tool = {
                'name': FINAL_RESULT_TOOL,
                'description': 'The final response which ends this conversation',
                'parameters_json_schema': schema,
                'kind': 'output',
            }
            params['output_tools'] = [output_tool]
            params['allow_text_output'] = False
            params['output_mode'] = 'tool'
        return params

    def _handle_response(self, state, response_dict):
        """Record the response; return the function tool calls to run, or set `state.output` when final."""
        response = ModelResponse([_load_part(p) for p in response_dict['parts']], response_dict)
        state.messages.append(response)
        state.usage['requests'] += 1
        for key in ('input_tokens', 'output_tokens'):
            state.usage[key] += response.usage().get(key) or 0
        calls = [p for p in response.parts if p.part_kind == 'tool-call']
        output_calls = [c for c in calls if c.tool_name == FINAL_RESULT_TOOL and self.output_type is not str]
        if output_calls:
            return self._handle_output_call(state, output_calls[0], calls)
        if calls:
            return calls
        text = ''.join([p.content for p in response.parts if p.part_kind == 'text'])
        if self.output_type is str:
            state.output = text
            state.done = True
            return []
        self._output_retry(state, RetryFeedbackPart('Please include your response in a tool call.', {}), 'no_output')
        return []

    def _handle_output_call(self, state, output_call, calls):
        output, errors = _build_output(self.output_type, output_call.args_as_dict())
        if errors:
            content = 'Validation errors:\n' + '\n'.join(errors) + '\n\nFix the errors and try again.'
            part = ToolReturnPart(FINAL_RESULT_TOOL, content, output_call.tool_call_id, 'retried')
            self._output_retry(state, part, 'validation_error')
            return []
        state.output = output
        state.done = True
        parts = []
        for call in calls:
            content = (
                'Final result processed.'
                if call is output_call
                else 'Tool not executed - a final result was already processed.'
            )
            parts.append(ToolReturnPart(call.tool_name, content, call.tool_call_id))
        self._finish_step(state, parts)
        return []

    def _output_retry(self, state, part, cause):
        if state.output_retries >= self.output_retries:
            raise RuntimeError(f'Exceeded maximum output retries ({self.output_retries}): {part.content}')
        state.output_retries += 1
        part.retry = {'cause': cause, 'attempt': state.output_retries, 'max_retries': self.output_retries}
        self._finish_step(state, [part])

    def _finish_step(self, state, parts):
        if parts:
            state.messages.append(ModelRequest(list(parts), self.instructions))

    def _result(self, state):
        return AgentRunResult(state.output, state.messages, state.new_start, state.usage)

    def _prepare_call(self, state, call):
        """Return `(tool, thunk)`, or `(None, retry_part)` for an unknown tool."""
        tool = self.tools.get(call.tool_name)
        if tool is None:
            available = ', '.join(self.tools) or 'none'
            msg = f'Unknown tool name: {call.tool_name!r}. Available tools: {available}'
            retry = {'cause': 'unknown_tool', 'attempt': None, 'max_retries': None}
            return None, ToolReturnPart(call.tool_name, msg, call.tool_call_id, 'retried', retry)
        ctx = RunContext(state.deps, state.messages, tool.name, state.retries.get(tool.name, 0), state.step)

        def thunk():
            args = call.args_as_dict()
            return tool.function(ctx, **args) if tool.takes_ctx else tool.function(**args)

        return tool, thunk

    def _tool_error(self, state, tool, call, error):
        attempt = state.retries.get(tool.name, 0) + 1
        if attempt > tool.max_retries:
            raise error
        state.retries[tool.name] = attempt
        retry = {'cause': 'model_retry', 'attempt': attempt, 'max_retries': tool.max_retries}
        content = f'{type(error).__name__}: {error}'
        return ToolReturnPart(tool.name, content, call.tool_call_id, 'retried', retry)

    def _call_tool(self, state, call):
        tool, thunk = self._prepare_call(state, call)
        if tool is None:
            return thunk
        try:
            result = thunk()
        except Exception as e:
            return self._tool_error(state, tool, call, e)
        if type(result).__name__ == 'coroutine':
            raise TypeError(f'Tool {tool.name!r} is async; use `await agent.run(...)`')
        return ToolReturnPart(tool.name, result, call.tool_call_id)

    async def _call_tool_async(self, state, call):
        tool, thunk = self._prepare_call(state, call)
        if tool is None:
            return thunk
        try:
            result = thunk()
            if type(result).__name__ == 'coroutine':
                result = await result
        except Exception as e:
            return self._tool_error(state, tool, call, e)
        return ToolReturnPart(tool.name, result, call.tool_call_id)
