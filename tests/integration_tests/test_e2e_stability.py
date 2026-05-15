"""Stability tests: repeated invocations produce structurally consistent output.

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


def test_repeated_runs_have_consistent_shape() -> None:
    tool = ApifyRunActorTool()
    results = [json.loads(tool.invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})) for _ in range(3)]

    for r in results:
        assert set(r.keys()) == {'run_id', 'status', 'dataset_id', 'started_at', 'finished_at'}
        assert r['status'] == 'SUCCEEDED'
    assert len({r['dataset_id'] for r in results}) == 3  # unique per run


def test_same_dataset_returns_identical_items() -> None:
    ds_id = json.loads(
        ApifyRunActorTool().invoke({'actor_id': _ACTOR_ID, 'run_input': _RUN_INPUT})
    )['dataset_id']

    tool = ApifyGetDatasetItemsTool()
    results = [json.loads(tool.invoke({'dataset_id': ds_id, 'limit': 10})) for _ in range(3)]

    assert all(r['items'] == results[0]['items'] for r in results)
