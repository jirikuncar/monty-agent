"""The device login and token refresh against a scripted Logfire authorization server."""

import json
import time
from collections.abc import Callable
from urllib.parse import parse_qs

import pytest

httpx = pytest.importorskip('httpx')

import oauth  # after the skip: the `tui` extra brings httpx
from oauth import Login

pytestmark = pytest.mark.anyio
BACKEND = 'https://logfire-eu.pydantic.dev'
TOKEN_URL = f'{BACKEND}/oauth/token'
RESOURCES = ['https://gateway-eu.pydantic.dev/proxy', BACKEND]


@pytest.fixture
def anyio_backend():
    return 'asyncio'


class Server:
    """Answers each endpoint from a queue of responses, recording the requests it got."""

    def __init__(self, tokens: list[httpx.Response]):
        self.tokens = tokens
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        match request.url.path:
            case '/.well-known/oauth-authorization-server':
                return httpx.Response(
                    200,
                    json={
                        'registration_endpoint': f'{BACKEND}/oauth/register',
                        'device_authorization_endpoint': f'{BACKEND}/oauth/device',
                        'token_endpoint': TOKEN_URL,
                    },
                )
            case '/oauth/register':
                return httpx.Response(201, json={'client_id': 'client-1'})
            case '/oauth/device':
                return httpx.Response(
                    200,
                    json={
                        'device_code': 'device-1',
                        'user_code': 'ABCD-EFGH',
                        'verification_uri': f'{BACKEND}/device',
                        'verification_uri_complete': f'{BACKEND}/device?code=ABCD-EFGH',
                        'expires_in': 600,
                        'interval': 2,
                    },
                )
            case '/oauth/token':
                return self.tokens.pop(0)
        return httpx.Response(404)

    def forms(self, path: str) -> list[dict[str, list[str]]]:
        return [parse_qs(r.content.decode()) for r in self.requests if r.url.path == path]


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[httpx.Response]], Server]:
    def start(tokens: list[httpx.Response]) -> Server:
        server = Server(tokens)
        client = httpx.AsyncClient
        monkeypatch.setattr(
            oauth.httpx, 'AsyncClient', lambda **kw: client(transport=httpx.MockTransport(server), **kw)
        )
        return server

    return start


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(oauth.asyncio, 'sleep', sleep)
    monkeypatch.setattr(oauth.webbrowser, 'open', lambda url: True)
    return waited


def pending(error: str) -> httpx.Response:
    return httpx.Response(400, json={'detail': {'error': error}})


def granted(access: str, refresh: str | None = 'refresh-1', expires_in: int = 1800) -> httpx.Response:
    body = {'access_token': access, 'expires_in': expires_in, 'token_type': 'Bearer'}
    return httpx.Response(200, json=body | ({'refresh_token': refresh} if refresh else {}))


async def test_device_login_polls_until_approved(serve, waits: list[float]):
    server = serve([pending('authorization_pending'), pending('slow_down'), granted('access-1', expires_in=60)])
    shown: list[tuple[str, str]] = []
    session = await oauth.login(BACKEND + '/', lambda url, code: shown.append((url, code)))

    assert shown == [(f'{BACKEND}/device?code=ABCD-EFGH', 'ABCD-EFGH')]
    assert waits == [2, 2, 7]  # slow_down adds five seconds to every later poll
    assert (session.backend, session.access_token, session.refresh_token) == (BACKEND, 'access-1', 'refresh-1')
    assert (session.client_id, session.token_endpoint) == ('client-1', TOKEN_URL)
    assert session.expires_at == pytest.approx(time.time() + 60, abs=5)

    registration = json.loads(next(r for r in server.requests if r.url.path == '/oauth/register').content)
    assert registration['scope'] == ' '.join(oauth.SCOPES)
    assert registration['token_endpoint_auth_method'] == 'none'
    [device] = server.forms('/oauth/device')
    assert device['resource'] == RESOURCES and device['code_challenge_method'] == ['S256']
    polls = server.forms('/oauth/token')
    assert len(polls) == 3 and all(p['device_code'] == ['device-1'] and 'code_verifier' in p for p in polls)


async def test_device_login_fails_on_denial(serve, waits: list[float]):
    serve([pending('access_denied')])
    with pytest.raises(RuntimeError, match='device flow failed.*access_denied'):
        await oauth.login(BACKEND, lambda url, code: None)


async def test_device_login_times_out(serve, monkeypatch: pytest.MonkeyPatch):
    server = serve([pending('authorization_pending')] * 1000)
    clock = [time.time()]

    async def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(oauth.asyncio, 'sleep', sleep)
    monkeypatch.setattr(oauth.time, 'time', lambda: clock[0])
    monkeypatch.setattr(oauth.webbrowser, 'open', lambda url: True)
    with pytest.raises(TimeoutError):
        await oauth.login(BACKEND, lambda url, code: None)
    assert len(server.forms('/oauth/token')) == 300  # every two seconds for the code's ten minutes


async def test_refresh_rotates_the_refresh_token(serve):
    server = serve([granted('access-2', refresh='refresh-2'), granted('access-3', refresh=None)])
    session = Login(BACKEND, 'access-1', 0, 'client-1', TOKEN_URL, 'refresh-1')

    renewed = await oauth.refresh(session)
    assert (renewed.access_token, renewed.refresh_token) == ('access-2', 'refresh-2')
    assert renewed.expires_at > time.time() + 1000
    again = await oauth.refresh(renewed)
    assert (again.access_token, again.refresh_token) == ('access-3', 'refresh-2')  # kept when none is issued

    first, second = server.forms('/oauth/token')
    assert first['grant_type'] == ['refresh_token'] and first['client_id'] == ['client-1']
    assert first['refresh_token'] == ['refresh-1'] and second['refresh_token'] == ['refresh-2']
    assert first['resource'] == RESOURCES


async def test_refresh_fails_when_spent_or_missing(serve):
    serve([httpx.Response(400, json={'error': 'invalid_grant'})])
    with pytest.raises(RuntimeError, match='token refresh failed \\(400\\).*invalid_grant'):
        await oauth.refresh(Login(BACKEND, 'a', 0, 'c', TOKEN_URL, 'spent'))
    with pytest.raises(RuntimeError, match='no refresh token'):
        await oauth.refresh(Login(BACKEND, 'a', 0, 'c', TOKEN_URL))


def test_env_points_every_client_at_the_backend_and_its_gateway():
    env = Login('https://logfire-us.pydantic.dev', 'tok', 0, 'c', TOKEN_URL).env()
    assert env['LOGFIRE_BASE_URL'] == 'https://logfire-us.pydantic.dev'
    assert env['PYDANTIC_AI_GATEWAY_BASE_URL'] == 'https://gateway-us.pydantic.dev/proxy'
    assert {env[k] for k in ('LOGFIRE_TOKEN', 'LOGFIRE_API_KEY', 'PYDANTIC_AI_GATEWAY_API_KEY')} == {'tok'}
