"""Auth failure tests: missing token and invalid token.

These do NOT require a valid APIFY_API_TOKEN.
"""

from __future__ import annotations

import json

import pytest

from langchain_apify import ApifyRunActorTool


def test_missing_token_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyRunActorTool()


def test_invalid_token_returns_error_string() -> None:
    tool = ApifyRunActorTool(apify_api_token='invalid_token_xyz_000')
    result = tool.invoke({'actor_id': 'apify/python-example', 'run_input': {}})
    assert isinstance(result, str)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)
