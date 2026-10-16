"""The integration marker in the user-agent has to survive apify-client upgrades.

`_create_apify_client` appends to the user-agent that apify-client builds inside its own HTTP client. Where that
header lives is not public API, so these tests run against the real client instead of a mock: if a future
apify-client moves it, they fail loudly. Without them the marker is dropped silently, which is exactly what
happened when this package moved from apify-client 2.x to 3.x.
"""

from __future__ import annotations

import pytest
from apify_client import ApifyClient, ApifyClientAsync

from langchain_apify._utils import _USER_AGENT_ORIGIN_SUFFIX, _create_apify_client

_CLIENT_CLASSES = [ApifyClient, ApifyClientAsync]


@pytest.mark.parametrize('client_cls', _CLIENT_CLASSES)
def test_create_apify_client_appends_origin_to_user_agent(client_cls: type) -> None:
    user_agent = _create_apify_client(client_cls, 'test-token').http_client._headers['User-Agent']

    # The client's own user-agent is extended, not replaced.
    assert user_agent.startswith('ApifyClient/')
    assert user_agent.endswith(_USER_AGENT_ORIGIN_SUFFIX)


@pytest.mark.parametrize('client_cls', _CLIENT_CLASSES)
def test_origin_survives_the_per_request_header_merge(client_cls: type) -> None:
    """Sitting on the client is not enough; the marker has to reach the headers of an actual request."""
    client = _create_apify_client(client_cls, 'test-token')

    headers, _, _ = client.http_client._prepare_request_call(headers=None, params=None)

    assert _USER_AGENT_ORIGIN_SUFFIX in headers['User-Agent']
