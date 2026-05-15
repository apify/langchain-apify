"""Tests for the two-step RunActor -> GetDatasetItems orchestration pattern."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from langchain_apify.tools import ApifyGetDatasetItemsTool, ApifyRunActorTool
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool


def test_dataset_id_flows_from_step1_to_step2(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.return_value = SUCCEEDED_RUN
    mock_tools_client.get_dataset_items.return_value = SAMPLE_ITEMS

    run_tool = make_tool(ApifyRunActorTool, mock_tools_client)
    items_tool = make_tool(ApifyGetDatasetItemsTool, mock_tools_client)

    dataset_id = json.loads(run_tool._run(actor_id='a/b', run_input={}))['dataset_id']
    parsed = json.loads(items_tool._run(dataset_id=dataset_id, limit=3))

    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.get_dataset_items.assert_called_once_with('dataset-xyz', 3, 0)


def test_failed_step1_is_not_valid_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.run_actor.side_effect = RuntimeError('Actor not found.')
    run_tool = make_tool(ApifyRunActorTool, mock_tools_client)

    result = run_tool.invoke({'actor_id': 'bad/actor'})

    with pytest.raises(json.JSONDecodeError):
        json.loads(result)


def test_each_run_produces_unique_dataset_id(mock_tools_client: MagicMock) -> None:
    runs = [{**SUCCEEDED_RUN, 'id': f'run-{i}', 'defaultDatasetId': f'ds-{i}'} for i in range(3)]
    mock_tools_client.run_actor.side_effect = runs
    run_tool = make_tool(ApifyRunActorTool, mock_tools_client)

    ids = [json.loads(run_tool._run(actor_id='a/b'))['dataset_id'] for _ in range(3)]

    assert ids == ['ds-0', 'ds-1', 'ds-2']
