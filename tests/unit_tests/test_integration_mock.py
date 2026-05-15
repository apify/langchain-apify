"""Tests for handle_tool_error via .invoke() and output schema contracts.

Existing test_tools.py tests ._run() directly. These test the LangChain
.invoke() path where ToolException becomes a string response, and verify
the exact output key sets match the documented contract.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_apify.tools import (
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyScrapeUrlTool,
)
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool


class TestHandleToolErrorViaInvoke:
    """ToolException must become a string (not raise) when using .invoke()."""

    def test_run_actor_error_becomes_string(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.run_actor.side_effect = RuntimeError('Actor not found.')
        tool = make_tool(ApifyRunActorTool, mock_tools_client)

        result = tool.invoke({'actor_id': 'bad/actor'})

        assert isinstance(result, str)
        assert 'Actor not found' in result

    def test_get_dataset_error_becomes_string(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.get_dataset_items.side_effect = RuntimeError('fetch failed')
        tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

        result = tool.invoke({'dataset_id': 'bad-id'})

        assert isinstance(result, str)
        assert 'fetch failed' in result

    def test_scrape_url_error_becomes_string(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.scrape_url.side_effect = RuntimeError('No content extracted.')
        tool = make_tool(ApifyScrapeUrlTool, mock_tools_client)

        result = tool.invoke({'url': 'https://bad.example.com'})

        assert isinstance(result, str)
        assert 'No content extracted' in result


class TestOutputSchemaContract:
    """Verify output key sets match the documented API contract."""

    _RUN_KEYS = {'run_id', 'status', 'dataset_id', 'started_at', 'finished_at'}

    def test_run_actor_keys(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
        tool = make_tool(ApifyRunActorTool, mock_tools_client)
        assert set(json.loads(tool._run(actor_id='a')).keys()) == self._RUN_KEYS

    def test_run_actor_and_get_dataset_keys(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.run_actor_and_get_items.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
        tool = make_tool(ApifyRunActorAndGetDatasetTool, mock_tools_client)
        result = json.loads(tool._run(actor_id='a'))
        assert set(result.keys()) == {'run', 'items'}
        assert set(result['run'].keys()) == self._RUN_KEYS

    def test_get_dataset_items_keys_nonempty(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.get_dataset_items.return_value = SAMPLE_ITEMS
        tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)
        result = json.loads(tool._run(dataset_id='ds'))
        assert set(result.keys()) == {'items'}

    def test_get_dataset_items_keys_empty(self, mock_tools_client: MagicMock) -> None:
        mock_tools_client.get_dataset_items.return_value = []
        tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)
        result = json.loads(tool._run(dataset_id='ds'))
        assert set(result.keys()) == {'items', 'message'}
