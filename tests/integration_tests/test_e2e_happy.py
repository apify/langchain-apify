"""E2E happy-path tests for all 6 core tools against the live Apify API.

Requires APIFY_API_TOKEN. Uses apify/python-example for fast, cheap runs.
"""

from __future__ import annotations

import json
import os

import pytest

from langchain_apify import (
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetDatasetTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
)

_ACTOR_ID = 'apify/python-example'
_RUN_INPUT = {'first_number': 2, 'second_number': 3}
_TASK_ID = os.getenv('APIFY_TASK_ID', 'dx_heroes/hello-world-task')
_RUN_KEYS = {'run_id', 'status', 'dataset_id', 'started_at', 'finished_at'}

pytestmark = pytest.mark.skipif(not os.getenv('APIFY_API_TOKEN'), reason='APIFY_API_TOKEN not set')


def test_run_actor(self) -> None:
    parsed = json.loads(ApifyRunActorTool().invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT}))
    assert set(parsed.keys()) == _RUN_KEYS
    assert parsed['status'] == 'SUCCEEDED'


def test_get_dataset_items() -> None:
    ds_id = json.loads(ApifyRunActorTool().invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT}))['dataset_id']
    parsed = json.loads(ApifyGetDatasetItemsTool().invoke({'dataset_id': ds_id, 'limit': 10}))
    assert isinstance(parsed['items'], list)


def test_run_actor_and_get_dataset() -> None:
    parsed = json.loads(ApifyRunActorAndGetDatasetTool().invoke(
        {'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT, 'dataset_items_limit': 10}
    ))
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert isinstance(parsed['items'], list)


def test_scrape_url() -> None:
    result = ApifyScrapeUrlTool().invoke({'url': 'https://crawlee.dev'})
    assert isinstance(result, str) and len(result) > 100


@pytest.mark.skipif(not _TASK_ID, reason='APIFY_TASK_ID not set')
def test_run_task() -> None:
    parsed = json.loads(ApifyRunTaskTool().invoke({'task_id': _TASK_ID}))
    assert parsed['status'] == 'SUCCEEDED'


@pytest.mark.skipif(not _TASK_ID, reason='APIFY_TASK_ID not set')
def test_run_task_and_get_dataset() -> None:
    parsed = json.loads(ApifyRunTaskAndGetDatasetTool().invoke({'task_id': _TASK_ID, 'dataset_items_limit': 5}))
    assert parsed['run']['status'] == 'SUCCEEDED'
