from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
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
    ApifyRunTaskAndGetItemsTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
    _ApifyGenericTool,
    _iso,
    _run_meta,
)
from langchain_apify.utils import actor_id_to_tool_name
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool

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
# _iso / _run_meta helpers
# ---------------------------------------------------------------------------


def test_iso_converts_datetime_to_string() -> None:
    dt = datetime(2025, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    assert _iso(dt) == '2025-06-15T12:30:45+00:00'


def test_iso_passes_through_string() -> None:
    assert _iso('2025-01-01T00:00:00.000Z') == '2025-01-01T00:00:00.000Z'


def test_iso_passes_through_none() -> None:
    assert _iso(None) is None


def test_run_meta_with_datetime_values_is_json_serializable() -> None:
    run = {
        'id': 'run-dt',
        'status': 'SUCCEEDED',
        'defaultDatasetId': 'ds-dt',
        'startedAt': datetime(2025, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
        'finishedAt': datetime(2025, 3, 1, 10, 1, 0, tzinfo=timezone.utc),
    }
    meta = _run_meta(run)
    serialized = json.dumps(meta)
    parsed = json.loads(serialized)
    assert parsed['run_id'] == 'run-dt'
    assert parsed['started_at'] == '2025-03-01T10:00:00+00:00'
    assert parsed['finished_at'] == '2025-03-01T10:01:00+00:00'


def test_run_meta_with_string_values_is_json_serializable() -> None:
    meta = _run_meta(SUCCEEDED_RUN)
    serialized = json.dumps(meta)
    parsed = json.loads(serialized)
    assert parsed['started_at'] == '2025-01-01T00:00:00.000Z'
    assert parsed['finished_at'] == '2025-01-01T00:01:00.000Z'


def test_run_meta_with_missing_timestamps() -> None:
    run = {'id': 'run-none', 'status': 'RUNNING', 'defaultDatasetId': 'ds-none'}
    meta = _run_meta(run)
    serialized = json.dumps(meta)
    parsed = json.loads(serialized)
    assert parsed['started_at'] is None
    assert parsed['finished_at'] is None


def test_run_actor_tool_with_datetime_run(mock_tools_client: MagicMock) -> None:
    """End-to-end: ApifyRunActorTool returns valid JSON when the client returns datetime objects."""
    mock_tools_client.run_actor.return_value = {
        'id': 'run-real',
        'status': 'SUCCEEDED',
        'defaultDatasetId': 'ds-real',
        'startedAt': datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        'finishedAt': datetime(2025, 6, 1, 8, 5, 0, tzinfo=timezone.utc),
    }
    tool = make_tool(ApifyRunActorTool, mock_tools_client)

    result = tool._run(actor_id='apify/test')

    parsed = json.loads(result)
    assert parsed['run_id'] == 'run-real'
    assert parsed['started_at'] == '2025-06-01T08:00:00+00:00'
    assert parsed['finished_at'] == '2025-06-01T08:05:00+00:00'


# ---------------------------------------------------------------------------
# ApifyRunActorTool
# ---------------------------------------------------------------------------


def test_run_actor_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunActorTool, mock_tools_client)

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
    tool = make_tool(ApifyRunActorTool, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(actor_id='apify/test')


def test_run_actor_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunActorTool()


# ---------------------------------------------------------------------------
# ApifyGetDatasetItemsTool
# ---------------------------------------------------------------------------


def test_get_dataset_items_tool_returns_json_object(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.return_value = SAMPLE_ITEMS
    tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    result = tool._run(dataset_id='dataset-xyz', limit=50, offset=5)

    parsed = json.loads(result)
    assert len(parsed['items']) == 2
    assert parsed['items'][0]['text'] == 'item-1'
    mock_tools_client.get_dataset_items.assert_called_once_with('dataset-xyz', 50, 5)


def test_get_dataset_items_tool_empty_returns_message(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.return_value = []
    tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    result = tool._run(dataset_id='dataset-empty')

    parsed = json.loads(result)
    assert parsed['items'] == []
    assert 'empty' in parsed['message'].lower()


def test_get_dataset_items_tool_network_error_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.side_effect = RuntimeError('Network error fetching dataset ds-bad: connection reset')
    tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    with pytest.raises(ToolException, match='Network error fetching dataset'):
        tool._run(dataset_id='ds-bad')


def test_get_dataset_items_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyGetDatasetItemsTool()


# ---------------------------------------------------------------------------
# ApifyRunActorAndGetItemsTool
# ---------------------------------------------------------------------------


def test_run_actor_and_get_items_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor_and_get_items.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyRunActorAndGetItemsTool, mock_tools_client)

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
    tool = make_tool(ApifyRunActorAndGetItemsTool, mock_tools_client)

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
    tool = make_tool(ApifyScrapeUrlTool, mock_tools_client)

    result = tool._run(url='https://example.com')

    assert result == '# Hello World'
    mock_tools_client.scrape_url.assert_called_once_with('https://example.com', 120)


def test_scrape_url_tool_empty_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.scrape_url.side_effect = RuntimeError('No content extracted from https://example.com.')
    tool = make_tool(ApifyScrapeUrlTool, mock_tools_client)

    with pytest.raises(ToolException, match='No content extracted'):
        tool._run(url='https://example.com')


def test_scrape_url_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyScrapeUrlTool()


# ---------------------------------------------------------------------------
# ApifyRunTaskTool
# ---------------------------------------------------------------------------


def test_run_task_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunTaskTool, mock_tools_client)

    result = tool._run(task_id='user/my-task', task_input={'key': 'val'})

    parsed = json.loads(result)
    assert parsed['run_id'] == 'run-abc'
    assert parsed['status'] == 'SUCCEEDED'
    assert parsed['dataset_id'] == 'dataset-xyz'
    assert parsed['started_at'] == '2025-01-01T00:00:00.000Z'
    assert parsed['finished_at'] == '2025-01-01T00:01:00.000Z'
    mock_tools_client.run_task.assert_called_once_with('user/my-task', {'key': 'val'}, 300, None)


def test_run_task_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task.side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = make_tool(ApifyRunTaskTool, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(task_id='user/my-task')


def test_run_task_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunTaskTool()


# ---------------------------------------------------------------------------
# ApifyRunTaskAndGetItemsTool
# ---------------------------------------------------------------------------


def test_run_task_and_get_items_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task_and_get_items.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyRunTaskAndGetItemsTool, mock_tools_client)

    result = tool._run(task_id='user/my-task', task_input={'q': '1'}, dataset_items_limit=50)

    parsed = json.loads(result)
    assert parsed['run']['run_id'] == 'run-abc'
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert len(parsed['items']) == 2
    mock_tools_client.run_task_and_get_items.assert_called_once_with('user/my-task', {'q': '1'}, 300, None, 50)


def test_run_task_and_get_items_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task_and_get_items.side_effect = RuntimeError(
        'Actor run run-bad ended with status TIMED-OUT.'
    )
    tool = make_tool(ApifyRunTaskAndGetItemsTool, mock_tools_client)

    with pytest.raises(ToolException, match='TIMED-OUT'):
        tool._run(task_id='user/my-task')


def test_run_task_and_get_items_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunTaskAndGetItemsTool()


# ---------------------------------------------------------------------------
# Value clamping (developer safety limits)
# ---------------------------------------------------------------------------


def test_run_actor_tool_clamps_timeout(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunActorTool, mock_tools_client, max_timeout_secs=60)

    tool._run(actor_id='apify/test', timeout_secs=9999)

    mock_tools_client.run_actor.assert_called_once_with('apify/test', None, 60, None)


def test_run_actor_tool_clamps_memory(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunActorTool, mock_tools_client, max_memory_mbytes=512)

    tool._run(actor_id='apify/test', memory_mbytes=8192)

    mock_tools_client.run_actor.assert_called_once_with('apify/test', None, 300, 512)


def test_run_actor_tool_passes_none_memory_through(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunActorTool, mock_tools_client, max_memory_mbytes=512)

    tool._run(actor_id='apify/test', memory_mbytes=None)

    mock_tools_client.run_actor.assert_called_once_with('apify/test', None, 300, None)


def test_get_dataset_items_tool_clamps_limit(mock_tools_client: MagicMock) -> None:
    mock_tools_client.get_dataset_items.return_value = SAMPLE_ITEMS
    tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client, max_items=10)

    tool._run(dataset_id='ds-1', limit=50000)

    mock_tools_client.get_dataset_items.assert_called_once_with('ds-1', 10, 0)


def test_run_actor_and_get_items_tool_clamps_all(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor_and_get_items.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(
        ApifyRunActorAndGetItemsTool,
        mock_tools_client,
        max_timeout_secs=30,
        max_memory_mbytes=256,
        max_items=5,
    )

    tool._run(actor_id='a', timeout_secs=9999, memory_mbytes=9999, dataset_items_limit=9999)

    mock_tools_client.run_actor_and_get_items.assert_called_once_with('a', None, 30, 256, 5)


def test_scrape_url_tool_clamps_timeout(mock_tools_client: MagicMock) -> None:
    mock_tools_client.scrape_url.return_value = '# content'
    tool = make_tool(ApifyScrapeUrlTool, mock_tools_client, max_timeout_secs=30)

    tool._run(url='https://example.com', timeout_secs=9999)

    mock_tools_client.scrape_url.assert_called_once_with('https://example.com', 30)


def test_run_task_tool_clamps_timeout_and_memory(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunTaskTool, mock_tools_client, max_timeout_secs=60, max_memory_mbytes=512)

    tool._run(task_id='t/1', timeout_secs=9999, memory_mbytes=9999)

    mock_tools_client.run_task.assert_called_once_with('t/1', None, 60, 512)


def test_run_task_and_get_items_tool_clamps_all(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_task_and_get_items.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(
        ApifyRunTaskAndGetItemsTool,
        mock_tools_client,
        max_timeout_secs=30,
        max_memory_mbytes=256,
        max_items=5,
    )

    tool._run(task_id='t/1', timeout_secs=9999, memory_mbytes=9999, dataset_items_limit=9999)

    mock_tools_client.run_task_and_get_items.assert_called_once_with('t/1', None, 30, 256, 5)


def test_values_below_max_pass_through(mock_tools_client: MagicMock) -> None:
    """When LLM values are within limits they should pass through unchanged."""
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    tool = make_tool(ApifyRunActorTool, mock_tools_client, max_timeout_secs=600, max_memory_mbytes=4096)

    tool._run(actor_id='apify/test', timeout_secs=120, memory_mbytes=1024)

    mock_tools_client.run_actor.assert_called_once_with('apify/test', None, 120, 1024)


# ---------------------------------------------------------------------------
# Tool metadata assertions
# ---------------------------------------------------------------------------


def test_generic_tools_have_correct_metadata() -> None:
    """Verify name, description, and args_schema are set on all generic tools."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tools = [
            ApifyRunActorTool(apify_api_token='dummy'),
            ApifyGetDatasetItemsTool(apify_api_token='dummy'),
            ApifyRunActorAndGetItemsTool(apify_api_token='dummy'),
            ApifyScrapeUrlTool(apify_api_token='dummy'),
            ApifyRunTaskTool(apify_api_token='dummy'),
            ApifyRunTaskAndGetItemsTool(apify_api_token='dummy'),
        ]

    expected_names = [
        'apify_run_actor',
        'apify_get_dataset_items',
        'apify_run_actor_and_get_items',
        'apify_scrape_url',
        'apify_run_task',
        'apify_run_task_and_get_items',
    ]

    for tool, expected_name in zip(tools, expected_names):
        assert tool.name == expected_name
        assert tool.description
        assert tool.args_schema is not None
        assert tool.handle_tool_error is True


# ---------------------------------------------------------------------------
# _ApifyGenericTool inheritance
# ---------------------------------------------------------------------------


def test_all_generic_tools_inherit_from_base() -> None:
    """Every generic tool must be a subclass of _ApifyGenericTool."""
    for tool_cls in (
        ApifyRunActorTool,
        ApifyGetDatasetItemsTool,
        ApifyRunActorAndGetItemsTool,
        ApifyScrapeUrlTool,
        ApifyRunTaskTool,
        ApifyRunTaskAndGetItemsTool,
    ):
        assert issubclass(tool_cls, _ApifyGenericTool), f'{tool_cls.__name__} must extend _ApifyGenericTool'


def test_legacy_tool_does_not_inherit_from_generic_base() -> None:
    """ApifyActorsTool is legacy and must NOT inherit from _ApifyGenericTool."""
    assert not issubclass(ApifyActorsTool, _ApifyGenericTool)


# ---------------------------------------------------------------------------
# APIFY_CORE_TOOLS list
# ---------------------------------------------------------------------------


def test_apify_core_tools_contains_all_generic_classes() -> None:
    """APIFY_CORE_TOOLS must list exactly the 6 generic tool classes."""
    assert set(APIFY_CORE_TOOLS) == {
        ApifyRunActorTool,
        ApifyGetDatasetItemsTool,
        ApifyRunActorAndGetItemsTool,
        ApifyScrapeUrlTool,
        ApifyRunTaskTool,
        ApifyRunTaskAndGetItemsTool,
    }
    assert len(APIFY_CORE_TOOLS) == 6
