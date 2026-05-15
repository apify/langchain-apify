"""Unit tests for clamping boundaries NOT covered by the existing test_tools.py.

Focuses on:
- _clamp_depth (not tested elsewhere)
- Exact at-boundary values (existing only tests above-max)
- Configurable thresholds with relaxed limits
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import ApifyRunActorTool, _ApifyGenericTool


def _make_tool(**kwargs) -> _ApifyGenericTool:  # type: ignore[type-arg]
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        return ApifyRunActorTool(apify_api_token='dummy', **kwargs)


class TestClampDepth:
    """_clamp_depth is not covered by existing tests at all."""

    @pytest.mark.parametrize('input_val,expected', [
        (-999, 0),
        (-1, 0),
        (0, 0),
        (3, 3),
        (5, 5),
        (100, 5),
    ])
    def test_boundaries(self, input_val: int, expected: int) -> None:
        tool = _make_tool(max_crawl_depth=5)
        assert tool._clamp_depth(input_val) == expected


class TestAtExactMax:
    """Existing tests only check above-max. Verify at-max passes through."""

    def test_timeout_at_max(self) -> None:
        assert _make_tool(max_timeout_secs=600)._clamp_timeout(600) == 600

    def test_memory_at_max(self) -> None:
        assert _make_tool(max_memory_mbytes=32768)._clamp_memory(32768) == 32768

    def test_items_at_max(self) -> None:
        assert _make_tool(max_items=1000)._clamp_items(1000) == 1000

    def test_depth_at_max(self) -> None:
        assert _make_tool(max_crawl_depth=5)._clamp_depth(5) == 5


class TestRelaxedLimits:
    """Verify relaxed custom limits allow higher values."""

    def test_high_limits_pass_through(self) -> None:
        tool = _make_tool(
            max_timeout_secs=9999,
            max_items=50000,
            max_memory_mbytes=65536,
            max_crawl_depth=20,
        )
        assert tool._clamp_timeout(5000) == 5000
        assert tool._clamp_items(30000) == 30000
        assert tool._clamp_memory(65536) == 65536
        assert tool._clamp_depth(15) == 15
