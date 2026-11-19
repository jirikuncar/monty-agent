"""Log in to Logfire without local state: register a public OAuth client (DCR), then run the device flow.

One access token covers the Agent Sandboxes, the AI Gateway, sending traces and querying them. It lives in memory only,
with the refresh token that renews it (`refresh`); refresh tokens rotate, so each one is used once.
"""

import asyncio
import base64
import hashlib
import secrets
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import httpx

DEVICE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'
FORM = {'Content-Type': 'application/x-www-form-urlencoded'}
SCOPES = (
    'project:execute_agent_sandboxes',
    'project:gateway_proxy',
    'project:write_otlp',
    'project:read',
    'project:read_otlp',
    'project:read_variables',  # the TUI layout; without these it is kept for the session only
    'project:write_variables',
)
# Gateway URL per Logfire backend; the Gateway accepts the backend's tokens.
GATEWAYS = {
    'https://logfire-eu.pydantic.info': 'https://gateway.pydantic.info',
    'https://logfire-us.pydantic.info': 'https://gateway.pydantic.info',
    'https://logfire-eu.pydantic.dev': 'https://gateway-eu.pydantic.dev',
    'https://logfire-us.pydantic.dev': 'https://gateway-us.pydantic.dev',
}


@dataclass
class Login:
    backend: str
    access_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    refresh_token: str | None = None
    gateway_url: str | None = None
    """The AI Gateway for a backend not in `GATEWAYS`, e.g. a self-hosted one; None there means no Gateway."""

    @property
    def gateway(self) -> str | None:
        return self.gateway_url or GATEWAYS.get(self.backend)

    def env(self) -> dict[str, str]:
        """Environment for `logfire.configure`, `logfire.agent_sandboxes` and the `gateway/...` models."""
        env = {
            'LOGFIRE_TOKEN': self.access_token,
            'LOGFIRE_BASE_URL': self.backend,
            'LOGFIRE_API_KEY': self.access_token,
        }
        if self.gateway is not None:
            env |= {
                'PYDANTIC_AI_GATEWAY_API_KEY': self.access_token,
                'PYDANTIC_AI_GATEWAY_BASE_URL': f'{self.gateway}/proxy',
            }
        return env


async def login(
    backend: str, show: Callable[[str, str], None], scopes: tuple[str, ...] = SCOPES, gateway: str | None = None
) -> Login:
    """`show(url, user_code)` tells the user where to approve; returns once they have."""
    backend = backend.rstrip('/')
    session = Login(backend, '', 0, '', '', gateway_url=gateway.rstrip('/') if gateway else None)
    scope = ' '.join(scopes)
    async with httpx.AsyncClient(timeout=30.0) as http:
        meta = (await http.get(f'{backend}/.well-known/oauth-authorization-server')).raise_for_status().json()
        registration = await http.post(
            meta['registration_endpoint'],
            json={
                'client_name': 'monty-agent',
                'grant_types': [DEVICE_GRANT, 'refresh_token'],
                'token_endpoint_auth_method': 'none',
                'scope': scope,
            },
        )
        client_id = registration.raise_for_status().json()['client_id']
        client = [('client_id', client_id), *resources(session)]

        verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        pkce = [('code_challenge', challenge), ('code_challenge_method', 'S256')]
        device = await http.post(
            meta['device_authorization_endpoint'],
            headers=FORM,
            content=urlencode([*client, ('scope', scope), *pkce]),
        )
        code = device.raise_for_status().json()
        url = code.get('verification_uri_complete') or code['verification_uri']
        show(url, code['user_code'])
        webbrowser.open(url)

        poll = [*client, ('grant_type', DEVICE_GRANT), ('device_code', code['device_code'])]
        poll.append(('code_verifier', verifier))
        interval = int(code.get('interval', 5))
        deadline = time.time() + int(code['expires_in'])
        while time.time() < deadline:
            await asyncio.sleep(interval)
            response = await http.post(meta['token_endpoint'], headers=FORM, content=urlencode(poll))
            if response.status_code == 200:
                return issued(
                    replace(session, client_id=client_id, token_endpoint=meta['token_endpoint']), response.json()
                )
            body = response.json()
            detail = body.get('detail', body)
            error = detail.get('error') if isinstance(detail, dict) else None
            if error == 'slow_down':
                interval += 5
            elif error != 'authorization_pending':
                raise RuntimeError(f'device flow failed ({response.status_code}): {response.text[:300]}')
    raise TimeoutError('device flow timed out')


async def refresh(session: Login) -> Login:
    """A new access token (and refresh token) without the user; fails once the refresh token is spent or revoked."""
    if session.refresh_token is None:
        raise RuntimeError('no refresh token: log in again')
    form = [('grant_type', 'refresh_token'), ('client_id', session.client_id), ('refresh_token', session.refresh_token)]
    # The token endpoint binds the resources to the refresh token; sending them again is harmless and portable.
    form += resources(session)
    async with httpx.AsyncClient(timeout=30.0) as http:
        response = await http.post(session.token_endpoint, headers=FORM, content=urlencode(form))
    if response.status_code != 200:
        raise RuntimeError(f'token refresh failed ({response.status_code}): {response.text[:300]}')
    return issued(session, response.json())


def resources(session: Login) -> list[tuple[str, str]]:
    gateway = [('resource', f'{session.gateway}/proxy')] if session.gateway else []
    return [*gateway, ('resource', session.backend)]


def issued(session: Login, token: dict[str, Any]) -> Login:
    return replace(
        session,
        access_token=token['access_token'],
        expires_at=time.time() + int(token.get('expires_in', 1800)),
        refresh_token=token.get('refresh_token', session.refresh_token),
    )
