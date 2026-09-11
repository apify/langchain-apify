"""Package-surface invariants for the merged tool families.

Ported from the offline ``test_merge_surface`` playground check: the core,
social, search, and transcript tool groups must keep their expected sizes,
stay disjoint, expose globally-unique tool ``name`` values (a name collision
would surface here), and keep ``_run_meta`` in a single canonical home.
"""

from __future__ import annotations

import pytest

from langchain_apify import (
    APIFY_CORE_TOOLS,
    APIFY_SEARCH_TOOLS,
    APIFY_SOCIAL_TOOLS,
    APIFY_TRANSCRIPT_TOOLS,
)


@pytest.mark.parametrize(
    ('group', 'expected'),
    [
        (APIFY_CORE_TOOLS, 6),
        (APIFY_SOCIAL_TOOLS, 7),
        (APIFY_SEARCH_TOOLS, 6),
        (APIFY_TRANSCRIPT_TOOLS, 3),
    ],
)
def test_tool_group_sizes(group: list[type], expected: int) -> None:
    assert len(group) == expected


def test_tool_groups_are_disjoint() -> None:
    assert not set(APIFY_SOCIAL_TOOLS) & set(APIFY_SEARCH_TOOLS)
    assert not set(APIFY_CORE_TOOLS) & set(APIFY_SOCIAL_TOOLS)
    assert not set(APIFY_CORE_TOOLS) & set(APIFY_SEARCH_TOOLS)
    assert not set(APIFY_TRANSCRIPT_TOOLS) & set(APIFY_CORE_TOOLS)
    assert not set(APIFY_TRANSCRIPT_TOOLS) & set(APIFY_SOCIAL_TOOLS)
    assert not set(APIFY_TRANSCRIPT_TOOLS) & set(APIFY_SEARCH_TOOLS)


def test_tool_names_are_globally_unique() -> None:
    all_tools = [*APIFY_CORE_TOOLS, *APIFY_SOCIAL_TOOLS, *APIFY_SEARCH_TOOLS, *APIFY_TRANSCRIPT_TOOLS]
    names = [tool_cls.model_fields['name'].default for tool_cls in all_tools]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f'duplicate tool names: {dupes}'


def test_run_meta_has_single_canonical_home() -> None:
    # _run_meta lives in tools.base; a stale duplicate in _utils would mean two
    # diverging copies after the tools/ package split.
    from langchain_apify.tools.base import _run_meta  # noqa: F401

    with pytest.raises(ImportError):
        from langchain_apify._utils import _run_meta as _stale  # type: ignore[attr-defined]  # noqa: F401
