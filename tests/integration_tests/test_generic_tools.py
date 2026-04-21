"""Integration smoke tests for the generic Apify tools.

These tests hit the real Apify API and require the ``APIFY_API_TOKEN``
environment variable to be set.  They use ``apify/python-example`` (a
trivial Actor that adds two numbers) to keep execution fast and cheap.
"""

from __future__ import annotations

import json
import os

import pytest

from langchain_apify import (
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetItemsTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetItemsTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
)

_ACTOR_ID = 'apify/python-example'
_RUN_INPUT = {'first_number': 2, 'second_number': 3}

pytestmark = pytest.mark.skipif(
    not os.getenv('APIFY_API_TOKEN'),
    reason='APIFY_API_TOKEN not set',
)


def test_run_actor_tool_smoke() -> None:
    tool = ApifyRunActorTool()
    result = tool.invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})

    parsed = json.loads(result)
    assert parsed['status'] == 'SUCCEEDED'
    assert parsed['run_id']
    assert parsed['dataset_id']


def test_get_dataset_items_tool_smoke() -> None:
    run_tool = ApifyRunActorTool()
    run_result = json.loads(run_tool.invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT}))
    dataset_id = run_result['dataset_id']

    items_tool = ApifyGetDatasetItemsTool()
    result = items_tool.invoke({'dataset_id': dataset_id, 'limit': 10})

    parsed = json.loads(result)
    assert 'items' in parsed
    assert isinstance(parsed['items'], list)


def test_run_actor_and_get_items_tool_smoke() -> None:
    tool = ApifyRunActorAndGetItemsTool()
    result = tool.invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})

    parsed = json.loads(result)
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert isinstance(parsed['items'], list)


def test_scrape_url_tool_smoke() -> None:
    tool = ApifyScrapeUrlTool()
    result = tool.invoke({'url': 'https://crawlee.dev'})

    assert isinstance(result, str)
    assert len(result) > 0


_TASK_ID = os.getenv('APIFY_TASK_ID', '')


@pytest.mark.skipif(not _TASK_ID, reason='APIFY_TASK_ID not set')
def test_run_task_tool_smoke() -> None:
    tool = ApifyRunTaskTool()
    result = tool.invoke({'task_id': _TASK_ID})

    parsed = json.loads(result)
    assert parsed['status'] == 'SUCCEEDED'
    assert parsed['run_id']
    assert parsed['dataset_id']


@pytest.mark.skipif(not _TASK_ID, reason='APIFY_TASK_ID not set')
def test_run_task_and_get_items_tool_smoke() -> None:
    tool = ApifyRunTaskAndGetItemsTool()
    result = tool.invoke({'task_id': _TASK_ID})

    parsed = json.loads(result)
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert isinstance(parsed['items'], list)
