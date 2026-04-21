from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import BaseModel

from langchain_apify import APIFY_CORE_TOOLS
from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import (
    ApifyActorsTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetItemsTool,
    ApifyRunActorTool,
    ApifyScrapeUrlTool,
)
from langchain_apify.utils import actor_id_to_tool_name

if TYPE_CHECKING:
    from collections.abc import Generator


def test_apify_actors_tool_instance() -> None:
    """Tests the ApifyActorsTool instance creation.

    Creates an instance of the ApifyActorsTool and
        checks if the instance is created correctly.
    """
    with (
        patch.object(
            ApifyActorsTool,
            '_create_description',
            return_value='Mocked description',
        ),
        patch.object(
            ApifyActorsTool,
            '_build_tool_args_schema_model',
        ) as mock_build_tool_args_schema_model,
    ):

        class DummyModel(BaseModel):
            run_input: str

        mock_build_tool_args_schema_model.return_value = DummyModel

        actor_id = 'apify/python-example'
        tool = ApifyActorsTool(actor_id=actor_id, apify_api_token='dummy-token')
        assert isinstance(tool, ApifyActorsTool)
        assert tool.description == 'Mocked description'
        assert tool.name == actor_id_to_tool_name(actor_id)
        assert tool.args_schema == DummyModel


def test_run_actor_method(apify_actors_tool_fixture: ApifyActorsTool) -> None:
    """Tests the ApifyActorsTool._run_actor method.

    Mocks the ApifyActorsTool._run_actor method to return a single item.
    """
    with patch.object(ApifyActorsTool, '_run_actor') as mock_run_actor:
        mock_run_actor.return_value = [{'text': 'Apify is great!'}]

        result = apify_actors_tool_fixture.invoke(
            input={'run_input': {'query': 'what is Apify?', 'maxResults': 3}},
        )
        mock_run_actor.assert_called_once()
        assert result[0]['text'] == 'Apify is great!'


@pytest.fixture
def apify_actors_tool_fixture() -> Generator[ApifyActorsTool, None, None]:
    """Fixture to create an instance of the ApifyActorsTool.

    Yields:
        ApifyActorsTool: An instance of the ApifyActorsTool.
    """
    with (
        patch.object(
            ApifyActorsTool,
            '_create_description',
            return_value='Mocked description',
        ),
        patch.object(
            ApifyActorsTool,
            '_build_tool_args_schema_model',
        ) as mock_build_tool_args_schema_model,
    ):

        class DummyModel(BaseModel):
            run_input: str | dict

        mock_build_tool_args_schema_model.return_value = DummyModel

        tool = ApifyActorsTool(actor_id='apify/python-example', apify_api_token='dummy-token')
        yield tool


# ---------------------------------------------------------------------------
# Shared test data for generic tools
# ---------------------------------------------------------------------------

_SUCCEEDED_RUN: dict = {
    'id': 'run-abc',
    'status': 'SUCCEEDED',
    'defaultDatasetId': 'dataset-xyz',
    'startedAt': '2025-01-01T00:00:00.000Z',
    'finishedAt': '2025-01-01T00:01:00.000Z',
}

_SAMPLE_ITEMS: list[dict] = [
    {'text': 'item-1', 'url': 'https://example.com/1'},
    {'text': 'item-2', 'url': 'https://example.com/2'},
]


@pytest.fixture
def mock_tools_client() -> MagicMock:
    return MagicMock(spec=ApifyToolsClient)


def _make_tool(tool_cls: type, mock_client: MagicMock) -> Any:  # noqa: ANN401
    """Instantiate a generic tool with a mocked ApifyToolsClient."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tool = tool_cls(apify_api_token='dummy-token')
    tool._client = mock_client
    return tool


# ---------------------------------------------------------------------------
# ApifyRunActorTool
# ---------------------------------------------------------------------------


def test_run_actor_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = _SUCCEEDED_RUN
    tool = _make_tool(ApifyRunActorTool, mock_tools_client)

    result = tool._run(actor_id='apify/test', run_input={'key': 'val'})

    parsed = json.loads(result)
    assert parsed['run_id'] == 'run-abc'
    assert parsed['status'] == 'SUCCEEDED'
    assert parsed['dataset_id'] == 'dataset-xyz'
    assert parsed['started_at'] == '2025-01-01T00:00:00.000Z'
    assert parsed['finished_at'] == '2025-01-01T00:01:00.000Z'
    mock_tools_client.run_actor.assert_called_once_with('apify/test', {'key': 'val'}, 300, None)


def test_run_actor_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = _make_tool(ApifyRunActorTool, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(actor_id='apify/test')


def test_run_actor_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunActorTool()


# ---------------------------------------------------------------------------
# ApifyGetDatasetItemsTool
# ---------------------------------------------------------------------------


def test_get_dataset_items_tool_returns_json_array(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.return_value = _SAMPLE_ITEMS
    tool = _make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    result = tool._run(dataset_id='dataset-xyz', limit=50, offset=5)

    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]['text'] == 'item-1'
    mock_tools_client.get_dataset_items.assert_called_once_with('dataset-xyz', 50, 5)


def test_get_dataset_items_tool_empty_returns_message(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.return_value = []
    tool = _make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    result = tool._run(dataset_id='dataset-empty')

    parsed = json.loads(result)
    assert parsed['items'] == []
    assert 'empty' in parsed['message'].lower()


def test_get_dataset_items_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyGetDatasetItemsTool()


# ---------------------------------------------------------------------------
# ApifyRunActorAndGetItemsTool
# ---------------------------------------------------------------------------


def test_run_actor_and_get_items_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor_and_get_items.return_value = (_SUCCEEDED_RUN, _SAMPLE_ITEMS)
    tool = _make_tool(ApifyRunActorAndGetItemsTool, mock_tools_client)

    result = tool._run(actor_id='apify/test', run_input={'q': '1'}, dataset_items_limit=50)

    parsed = json.loads(result)
    assert parsed['run']['run_id'] == 'run-abc'
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert len(parsed['items']) == 2
    mock_tools_client.run_actor_and_get_items.assert_called_once_with('apify/test', {'q': '1'}, 300, None, 50)


def test_run_actor_and_get_items_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor_and_get_items.side_effect = RuntimeError(
        'Actor run run-bad ended with status TIMED-OUT.'
    )
    tool = _make_tool(ApifyRunActorAndGetItemsTool, mock_tools_client)

    with pytest.raises(ToolException, match='TIMED-OUT'):
        tool._run(actor_id='apify/test')


def test_run_actor_and_get_items_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunActorAndGetItemsTool()


# ---------------------------------------------------------------------------
# ApifyScrapeUrlTool
# ---------------------------------------------------------------------------


def test_scrape_url_tool_returns_markdown(mock_tools_client: MagicMock) -> None:
    mock_tools_client.scrape_url.return_value = '# Hello World'
    tool = _make_tool(ApifyScrapeUrlTool, mock_tools_client)

    result = tool._run(url='https://example.com')

    assert result == '# Hello World'
    mock_tools_client.scrape_url.assert_called_once_with('https://example.com', 120)


def test_scrape_url_tool_empty_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.scrape_url.side_effect = RuntimeError('No content extracted from https://example.com.')
    tool = _make_tool(ApifyScrapeUrlTool, mock_tools_client)

    with pytest.raises(ToolException, match='No content extracted'):
        tool._run(url='https://example.com')


def test_scrape_url_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyScrapeUrlTool()


# ---------------------------------------------------------------------------
# Tool metadata assertions
# ---------------------------------------------------------------------------


def test_generic_tools_have_correct_metadata() -> None:
    """Verify name, description, and args_schema are set on all 4 tools."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tools = [
            ApifyRunActorTool(apify_api_token='dummy'),
            ApifyGetDatasetItemsTool(apify_api_token='dummy'),
            ApifyRunActorAndGetItemsTool(apify_api_token='dummy'),
            ApifyScrapeUrlTool(apify_api_token='dummy'),
        ]

    expected_names = [
        'apify_run_actor',
        'apify_get_dataset_items',
        'apify_run_actor_and_get_items',
        'apify_scrape_url',
    ]

    for tool, expected_name in zip(tools, expected_names):
        assert tool.name == expected_name
        assert tool.description
        assert tool.args_schema is not None
        assert tool.handle_tool_error is True


# ---------------------------------------------------------------------------
# APIFY_CORE_TOOLS list
# ---------------------------------------------------------------------------


def test_apify_core_tools_contains_all_four_classes() -> None:
    """APIFY_CORE_TOOLS must list exactly the 4 generic tool classes."""
    assert set(APIFY_CORE_TOOLS) == {
        ApifyRunActorTool,
        ApifyGetDatasetItemsTool,
        ApifyRunActorAndGetItemsTool,
        ApifyScrapeUrlTool,
    }
    assert len(APIFY_CORE_TOOLS) == 4
