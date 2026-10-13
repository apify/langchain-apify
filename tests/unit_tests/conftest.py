from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apify_client._models import Run
from apify_client._resource_clients.dataset import DatasetItemsPage

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


def make_run(**overrides: Any) -> Run:  # noqa: ANN401
    """Build a ``Run`` from the shared payload, overriding camelCase API fields."""
    return Run.model_validate({**_RUN_PAYLOAD, **overrides})


SUCCEEDED_RUN = make_run()

FAILED_RUN = make_run(id='run-fail', status='FAILED')

RUNNING_RUN = make_run(id='run-running', status='RUNNING', finishedAt=None)

SAMPLE_ITEMS: list[dict] = [
    {'text': 'item-1', 'url': 'https://example.com/1'},
    {'text': 'item-2', 'url': 'https://example.com/2'},
]


def make_items_page(items: list[dict], *, limit: int = 1000) -> DatasetItemsPage:
    """Build the page that ``DatasetClient.list_items`` returns."""
    return DatasetItemsPage(items=items, total=len(items), offset=0, count=len(items), limit=limit, desc=False)


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
