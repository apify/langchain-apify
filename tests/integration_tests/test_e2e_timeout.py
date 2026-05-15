"""Timeout and threshold tests against the live Apify API.

Requires APIFY_API_TOKEN.
"""

from __future__ import annotations

import json
import os

import pytest

from langchain_apify import ApifyGetDatasetItemsTool, ApifyRunActorTool

_ACTOR_ID = 'apify/python-example'
_RUN_INPUT = {'first_number': 2, 'second_number': 3}

pytestmark = pytest.mark.skipif(not os.getenv('APIFY_API_TOKEN'), reason='APIFY_API_TOKEN not set')


def test_offset_beyond_dataset_returns_empty() -> None:
    ds_id = json.loads(
        ApifyRunActorTool().invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})
    )['dataset_id']

    result = json.loads(ApifyGetDatasetItemsTool().invoke({'dataset_id': ds_id, 'limit': 10, 'offset': 999999}))
    assert result['items'] == []


def test_limit_larger_than_dataset_returns_available() -> None:
    ds_id = json.loads(
        ApifyRunActorTool().invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})
    )['dataset_id']

    result = json.loads(ApifyGetDatasetItemsTool().invoke({'dataset_id': ds_id, 'limit': 1000}))
    assert isinstance(result['items'], list)


def test_memory_128mb_accepted() -> None:
    parsed = json.loads(ApifyRunActorTool().invoke({
        'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT, 'memory_mbytes': 128,
    }))
    assert parsed['status'] == 'SUCCEEDED'
