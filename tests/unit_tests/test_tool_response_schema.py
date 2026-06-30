from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from langchain_apify import (
    APIFY_CORE_TOOLS,
    APIFY_SEARCH_TOOLS,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetDatasetTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
)
from langchain_apify.tools import (
    ApifyEcommerceScraperTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyRAGWebBrowserTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
)
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool


def _assert_envelope_shape(payload: dict) -> None:
    assert set(payload) == {'run', 'items'}
    assert isinstance(payload['items'], list)
    assert payload['run'] is None or isinstance(payload['run'], dict)


@pytest.mark.parametrize(
    ('tool_cls', 'setup_method', 'run_kwargs'),
    [
        (ApifyRunActorTool, 'run_actor', {'actor_id': 'apify/test'}),
        (ApifyGetDatasetItemsTool, 'get_dataset_items', {'dataset_id': 'dataset-xyz'}),
        (ApifyRunActorAndGetDatasetTool, 'run_actor_and_get_items', {'actor_id': 'apify/test'}),
        (ApifyScrapeUrlTool, 'scrape_url_with_metadata', {'url': 'https://example.com'}),
        (ApifyRunTaskTool, 'run_task', {'task_id': 'user/my-task'}),
        (ApifyRunTaskAndGetDatasetTool, 'run_task_and_get_items', {'task_id': 'user/my-task'}),
        (ApifyGoogleSearchTool, 'google_search', {'query': 'apify'}),
        (ApifyWebCrawlerTool, 'crawl_website', {'url': 'https://example.com'}),
        (ApifyRAGWebBrowserTool, 'rag_web_search', {'query': 'langchain'}),
        (ApifyGoogleMapsTool, 'google_maps_search', {'query': 'coffee'}),
        (ApifyYouTubeScraperTool, 'youtube_scrape', {'search_query': 'langchain'}),
        (ApifyEcommerceScraperTool, 'ecommerce_scrape', {'url': 'https://shop.example.com/p/1'}),
    ],
)
def test_all_tools_return_normalized_envelope(
    mock_tools_client: MagicMock, tool_cls: type, setup_method: str, run_kwargs: dict
) -> None:
    if setup_method in {'run_actor', 'run_task'}:
        getattr(mock_tools_client, setup_method).return_value = SUCCEEDED_RUN
    elif setup_method == 'get_dataset_items':
        getattr(mock_tools_client, setup_method).return_value = SAMPLE_ITEMS
    elif setup_method == 'scrape_url_with_metadata':
        mock_tools_client.scrape_url_with_metadata.return_value = (
            SUCCEEDED_RUN,
            [{'url': 'https://example.com', 'markdown': '# content'}],
            '# content',
            'markdown',
        )
    elif setup_method in {'run_actor_and_get_items', 'run_task_and_get_items'}:
        getattr(mock_tools_client, setup_method).return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    elif setup_method == 'google_search':
        mock_tools_client.google_search.return_value = (
            SUCCEEDED_RUN,
            [{'title': 'A', 'url': 'https://a', 'description': 'd'}],
        )
    elif setup_method == 'crawl_website':
        mock_tools_client.crawl_website.return_value = (
            SUCCEEDED_RUN,
            [{'url': 'https://example.com', 'markdown': '# Home', 'metadata': {'title': 'Home'}}],
        )
    elif setup_method == 'rag_web_search':
        mock_tools_client.rag_web_search.return_value = (
            SUCCEEDED_RUN,
            [{'crawledUrl': 'https://example.com', 'metadata': {'title': 'Home'}, 'text': 'Home'}],
        )
    elif setup_method in {'google_maps_search', 'youtube_scrape', 'ecommerce_scrape'}:
        getattr(mock_tools_client, setup_method).return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)

    tool = make_tool(tool_cls, mock_tools_client)
    payload = json.loads(tool._run(**run_kwargs))
    _assert_envelope_shape(payload)


def test_empty_result_is_normalized(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_maps_search.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyGoogleMapsTool, mock_tools_client)

    payload = json.loads(tool._run(query='empty'))
    _assert_envelope_shape(payload)
    assert payload['items'] == []


def test_tool_group_lists_cover_all_normalized_tools() -> None:
    assert set(APIFY_CORE_TOOLS) == {
        ApifyRunActorTool,
        ApifyGetDatasetItemsTool,
        ApifyRunActorAndGetDatasetTool,
        ApifyScrapeUrlTool,
        ApifyRunTaskTool,
        ApifyRunTaskAndGetDatasetTool,
    }
    assert set(APIFY_SEARCH_TOOLS) == {
        ApifyGoogleSearchTool,
        ApifyWebCrawlerTool,
        ApifyRAGWebBrowserTool,
        ApifyGoogleMapsTool,
        ApifyYouTubeScraperTool,
        ApifyEcommerceScraperTool,
    }
