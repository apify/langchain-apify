from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apify_client._models import Run

from langchain_apify._client import ApifyToolsClient

# Shaped like a real ``GET /v2/actor-runs/{id}`` response and validated into the
# ``Run`` model that ``.call()`` returns, so any dict-style access fails the tests.
_RUN_PAYLOAD: dict = {
    'id': 'run-abc',
    'actId': 'actor-abc',
    'userId': 'user-abc',
    'status': 'SUCCEEDED',
    'startedAt': '2025-01-01T00:00:00.000Z',
    'finishedAt': '2025-01-01T00:01:00.000Z',
    'buildId': 'build-xyz',
    'defaultDatasetId': 'dataset-xyz',
    'defaultKeyValueStoreId': 'kvs-xyz',
    'defaultRequestQueueId': 'queue-xyz',
    'meta': {'origin': 'API'},
    'stats': {'computeUnits': 0.01, 'restartCount': 0, 'resurrectCount': 0},
    'options': {'build': 'latest', 'timeoutSecs': 300, 'memoryMbytes': 1024, 'diskMbytes': 2048},
}

SUCCEEDED_RUN = Run.model_validate(_RUN_PAYLOAD)

FAILED_RUN = Run.model_validate({**_RUN_PAYLOAD, 'id': 'run-fail', 'status': 'FAILED'})

SAMPLE_ITEMS: list[dict] = [
    {'text': 'item-1', 'url': 'https://example.com/1'},
    {'text': 'item-2', 'url': 'https://example.com/2'},
]


@pytest.fixture
def mock_tools_client() -> MagicMock:
    return MagicMock(spec=ApifyToolsClient)


@pytest.fixture
def mock_apify_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(mock_apify_client: MagicMock) -> ApifyToolsClient:
    with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client):
        return ApifyToolsClient(apify_token='dummy-token')


def make_tool(tool_cls: type, mock_client: MagicMock, **kwargs: Any) -> Any:  # noqa: ANN401
    """Instantiate a generic tool with a mocked ApifyToolsClient."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tool = tool_cls(apify_token='dummy-token', **kwargs)
    tool._client = mock_client
    return tool
