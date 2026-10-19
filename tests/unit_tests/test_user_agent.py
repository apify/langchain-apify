"""The tracking headers that attribute a run to this integration have to actually be sent.

`_create_apify_client` sets two: `x-apify-integration-platform`, which the Apify platform reads, and a suffix on
the user-agent that apify-client builds inside its own HTTP client. Neither has a public accessor, so these tests
run against the real client instead of a mock: if a future apify-client moves where headers live, they fail
loudly. Without them a marker is dropped silently, which is exactly what happened when this package moved from
apify-client 2.x to 3.x.
"""

from __future__ import annotations

import warnings

import pytest
from apify_client import ApifyClient, ApifyClientAsync

from langchain_apify._utils import (
    _INTEGRATION_PLATFORM_HEADER,
    _USER_AGENT_ORIGIN_SUFFIX,
    _create_apify_client,
)

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


@pytest.mark.parametrize('client_cls', _CLIENT_CLASSES)
def test_integration_platform_header_reaches_the_request(client_cls: type) -> None:
    """This is the header the Apify platform reads; the user-agent marker predates it."""
    client = _create_apify_client(client_cls, 'test-token')

    headers, _, _ = client.http_client._prepare_request_call(headers=None, params=None)

    assert headers['x-apify-integration-platform'] == 'langchain'
    # Setting it must not cost us the defaults apify-client builds.
    assert headers['User-Agent'].startswith('ApifyClient/')
    assert headers['Authorization'] == 'Bearer test-token'


def test_integration_platform_header_raises_no_warning() -> None:
    """apify-client warns when a custom header overrides one of its defaults; this one must not be such a header."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _create_apify_client(ApifyClient, 'test-token')


def test_integration_platform_header_constant_is_not_mutated() -> None:
    """The constant is shared by every client in the package, so apify-client must not be able to write to it."""
    before = dict(_INTEGRATION_PLATFORM_HEADER)

    client = _create_apify_client(ApifyClient, 'test-token')
    client.http_client._prepare_request_call(headers=None, params=None)

    assert before == _INTEGRATION_PLATFORM_HEADER
