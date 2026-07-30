"""Tests for the clamp-ceiling text in tool input-schema descriptions.

``_ApifyGenericTool`` silently clamps ``timeout_secs`` / ``memory_mbytes``
/ ``limit`` / ``dataset_items_limit`` / ``max_crawl_depth`` / ``max_results``
/ ``max_crawl_pages`` to the per-tool ``max_*`` ceilings. The Pydantic input
schemas advertise those ceilings in their ``Field(description=...)`` strings
so an LLM agent doesn't promise results above the cap. These tests pin
every clamp-relevant field across the core, search, and social tools:

  1. every clamp-relevant Field carries a ``"clamped to N max"`` phrase;
  2. ``N`` matches the live default cap on ``_ApifyGenericTool`` (no
     drift between the description text and the actual clamp).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from langchain_apify.tools.base import _ApifyGenericTool

if TYPE_CHECKING:
    from pydantic import BaseModel
from langchain_apify.tools.core import (
    ApifyGetDatasetItemsInput,
    ApifyRunActorAndGetDatasetInput,
    ApifyRunActorInput,
    ApifyRunTaskAndGetDatasetInput,
    ApifyRunTaskInput,
    ApifyScrapeUrlInput,
)
from langchain_apify.tools.search import (
    ApifyEcommerceScraperInput,
    ApifyGoogleMapsInput,
    ApifyGoogleSearchInput,
    ApifyRAGWebBrowserInput,
    ApifyRAGWebBrowserTool,
    ApifyWebCrawlerInput,
    ApifyYouTubeScraperInput,
)
from langchain_apify.tools.social import (
    ApifyFacebookPostsScraperInput,
    ApifyInstagramScraperInput,
    ApifyLinkedInProfilePostsInput,
    ApifyLinkedInProfileSearchInput,
    ApifyTikTokScraperInput,
    ApifyTwitterScraperInput,
)

# (schema, field_name, base-class cap-field name)
_CLAMP_FIELDS: list[tuple[type[BaseModel], str, str]] = [
    (ApifyRunActorInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyRunActorInput, 'memory_mbytes', 'max_memory_mbytes'),
    (ApifyGetDatasetItemsInput, 'limit', 'max_items'),
    (ApifyRunActorAndGetDatasetInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyRunActorAndGetDatasetInput, 'memory_mbytes', 'max_memory_mbytes'),
    (ApifyRunActorAndGetDatasetInput, 'dataset_items_limit', 'max_items'),
    (ApifyScrapeUrlInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyGoogleSearchInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyWebCrawlerInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyWebCrawlerInput, 'max_crawl_depth', 'max_crawl_depth'),
    (ApifyRunTaskInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyRunTaskInput, 'memory_mbytes', 'max_memory_mbytes'),
    (ApifyRunTaskAndGetDatasetInput, 'timeout_secs', 'max_timeout_secs'),
    (ApifyRunTaskAndGetDatasetInput, 'memory_mbytes', 'max_memory_mbytes'),
    (ApifyRunTaskAndGetDatasetInput, 'dataset_items_limit', 'max_items'),
    # max_results / max_crawl_pages are clamped via _clamp_items (max_items cap).
    (ApifyGoogleSearchInput, 'max_results', 'max_items'),
    (ApifyWebCrawlerInput, 'max_crawl_pages', 'max_items'),
    (ApifyRAGWebBrowserInput, 'max_results', 'max_items'),
    (ApifyGoogleMapsInput, 'max_results', 'max_items'),
    (ApifyYouTubeScraperInput, 'max_results', 'max_items'),
    (ApifyEcommerceScraperInput, 'max_results', 'max_items'),
    (ApifyInstagramScraperInput, 'max_results', 'max_items'),
    (ApifyLinkedInProfilePostsInput, 'max_results', 'max_items'),
    (ApifyLinkedInProfileSearchInput, 'max_results', 'max_items'),
    (ApifyTwitterScraperInput, 'max_results', 'max_items'),
    (ApifyTikTokScraperInput, 'max_results', 'max_items'),
    (ApifyFacebookPostsScraperInput, 'max_results', 'max_items'),
]

# Most caps live on the _ApifyGenericTool base class. Tools that override a
# cap for their Actor advertise their own ceiling, so their description must be
# validated against the override rather than the generic default.
_CAP_SOURCE: dict[type[BaseModel], type[BaseModel]] = {
    ApifyRAGWebBrowserInput: ApifyRAGWebBrowserTool,
}


def _cap_default(schema: type[BaseModel], cap_field: str) -> int:
    """Return the live clamp ceiling that ``schema``'s description must match."""
    source = _CAP_SOURCE.get(schema, _ApifyGenericTool)
    return source.model_fields[cap_field].default


_CAP_PATTERN = re.compile(r'clamped to (\d+) max')


@pytest.mark.parametrize(('schema', 'field', 'cap_field'), _CLAMP_FIELDS)
def test_field_description_carries_cap_text(schema: type[BaseModel], field: str, cap_field: str) -> None:
    """Every clamp-relevant Field description ends with ``(clamped to N max)``."""
    description = schema.model_fields[field].description or ''
    expected_cap = _cap_default(schema, cap_field)
    assert f'clamped to {expected_cap} max' in description, (
        f'{schema.__name__}.{field} description does not mention the cap '
        f'(expected "clamped to {expected_cap} max"): {description!r}'
    )


@pytest.mark.parametrize(('schema', 'field', 'cap_field'), _CLAMP_FIELDS)
def test_field_description_cap_matches_base_class_default(schema: type[BaseModel], field: str, cap_field: str) -> None:
    """The number in the description text equals the live ``_ApifyGenericTool`` cap.

    Catches drift where a cap default moves but the description string is
    forgotten, or vice versa.
    """
    description = schema.model_fields[field].description or ''
    match = _CAP_PATTERN.search(description)
    assert match is not None, f'no "clamped to N max" phrase in {schema.__name__}.{field}: {description!r}'

    advertised_cap = int(match.group(1))
    actual_cap = _cap_default(schema, cap_field)
    assert advertised_cap == actual_cap, (
        f'{schema.__name__}.{field} advertises cap={advertised_cap} but '
        f'_ApifyGenericTool.{cap_field} default is {actual_cap}'
    )
