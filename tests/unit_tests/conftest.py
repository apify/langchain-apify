from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_apify._client import ApifyToolsClient

SUCCEEDED_RUN: dict = {
    'id': 'run-abc',
    'status': 'SUCCEEDED',
    'defaultDatasetId': 'dataset-xyz',
    'startedAt': '2025-01-01T00:00:00.000Z',
    'finishedAt': '2025-01-01T00:01:00.000Z',
}

FAILED_RUN: dict = {
    'id': 'run-fail',
    'status': 'FAILED',
    'defaultDatasetId': 'dataset-xyz',
}

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
        return ApifyToolsClient(apify_api_token='dummy-token')


def make_tool(tool_cls: type, mock_client: MagicMock, **kwargs: Any) -> Any:  # noqa: ANN401
    """Instantiate a generic tool with a mocked ApifyToolsClient."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tool = tool_cls(apify_api_token='dummy-token', **kwargs)
    tool._client = mock_client
    return tool
