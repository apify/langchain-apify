"""E2E negative tests: invalid IDs produce graceful error strings.

Requires APIFY_API_TOKEN.
"""

from __future__ import annotations

import json
import os

import pytest

from langchain_apify import (
    ApifyGetDatasetItemsTool,
    ApifyRunActorTool,
    ApifyRunTaskTool,
)

pytestmark = pytest.mark.skipif(not os.getenv('APIFY_API_TOKEN'), reason='APIFY_API_TOKEN not set')


@pytest.mark.parametrize('tool_cls,invoke_args', [
    (ApifyRunActorTool, {'actor_id': 'definitely/not-a-real-actor'}),
    (ApifyGetDatasetItemsTool, {'dataset_id': 'definitely-not-a-real-dataset-id'}),
    (ApifyRunTaskTool, {'task_id': 'definitely/not-a-real-task'}),
])
def test_invalid_id_returns_error_string(tool_cls, invoke_args) -> None:
    """handle_tool_error=True ensures no exception propagates; returns a string."""
    result = tool_cls().invoke(invoke_args)
    assert isinstance(result, str) and len(result) > 0
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)
