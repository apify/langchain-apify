from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from langchain_apify import (
    APIFY_ACTOR_TOOLS,
    APIFY_SEARCH_TOOLS,
    ApifyEcommerceScraperTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyRAGWebBrowserTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
)
from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import _ApifyGenericTool
from tests.unit_tests.conftest import SUCCEEDED_RUN, make_tool

# ---------------------------------------------------------------------------
# ApifyGoogleSearchTool
# ---------------------------------------------------------------------------


def test_google_search_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.return_value = [
        {'title': 'Result 1', 'url': 'https://example.com/1', 'description': 'Desc 1'},
        {'title': 'Result 2', 'url': 'https://example.com/2', 'description': 'Desc 2'},
    ]
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    result = tool._run(query='test query')

    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]['title'] == 'Result 1'
    assert parsed[1]['url'] == 'https://example.com/2'


def test_google_search_tool_passes_params(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    tool._run(query='test', max_results=5, country_code='us', language_code='en', timeout_secs=120)

    mock_tools_client.google_search.assert_called_once_with(
        'test',
        max_results=5,
        country_code='us',
        language_code='en',
        timeout_secs=120,
    )


def test_google_search_tool_clamps_timeout(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client, max_timeout_secs=60)

    tool._run(query='test', timeout_secs=9999)

    assert mock_tools_client.google_search.call_args.kwargs['timeout_secs'] == 60


def test_google_search_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client, max_items=3)

    tool._run(query='test', max_results=100)

    call_kwargs = mock_tools_client.google_search.call_args
    assert call_kwargs.kwargs['max_results'] == 3


def test_google_search_tool_empty_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    result = tool._run(query='nothing')

    assert json.loads(result) == []


def test_google_search_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.google_search.side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(query='test')


def test_google_search_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyGoogleSearchTool()


@pytest.mark.parametrize('bad_code', ['USA', 'english', 'u', 'us1', ''])
def test_google_search_tool_rejects_malformed_locale(mock_tools_client: MagicMock, bad_code: str) -> None:
    """country_code and language_code must be exactly two letters."""
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    with pytest.raises(ValueError, match='string_pattern_mismatch|String should match pattern'):
        tool.invoke({'query': 'test', 'country_code': bad_code})

    with pytest.raises(ValueError, match='string_pattern_mismatch|String should match pattern'):
        tool.invoke({'query': 'test', 'language_code': bad_code})


@pytest.mark.parametrize('raw_country', ['us', 'US', 'Us', 'uS'])
def test_google_search_tool_normalises_country_code_to_lower(mock_tools_client: MagicMock, raw_country: str) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    tool.invoke({'query': 'test', 'country_code': raw_country})

    assert mock_tools_client.google_search.call_args.kwargs['country_code'] == 'us'


@pytest.mark.parametrize('raw_language', ['en', 'EN', 'En', 'eN'])
def test_google_search_tool_normalises_language_code_to_lower(mock_tools_client: MagicMock, raw_language: str) -> None:
    mock_tools_client.google_search.return_value = []
    tool = make_tool(ApifyGoogleSearchTool, mock_tools_client)

    tool.invoke({'query': 'test', 'language_code': raw_language})

    assert mock_tools_client.google_search.call_args.kwargs['language_code'] == 'en'


# ---------------------------------------------------------------------------
# ApifyWebCrawlerTool
# ---------------------------------------------------------------------------


def test_web_crawler_tool_returns_json(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.return_value = [
        {'url': 'https://example.com/', 'markdown': '# Home', 'text': 'Home', 'metadata': {'title': 'Home'}},
        {'url': 'https://example.com/about', 'markdown': '', 'text': 'About us', 'metadata': {'title': 'About'}},
    ]
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client)

    result = tool._run(url='https://example.com')

    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0] == {'url': 'https://example.com/', 'title': 'Home', 'content': '# Home'}
    assert parsed[1] == {'url': 'https://example.com/about', 'title': 'About', 'content': 'About us'}


def test_web_crawler_tool_passes_params(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.return_value = []
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client)

    tool._run(
        url='https://example.com',
        max_crawl_pages=5,
        max_crawl_depth=2,
        crawler_type='playwright:firefox',
        timeout_secs=120,
    )

    mock_tools_client.crawl_website.assert_called_once_with(
        'https://example.com',
        max_crawl_pages=5,
        max_crawl_depth=2,
        crawler_type='playwright:firefox',
        timeout_secs=120,
    )


def test_web_crawler_tool_clamps_pages_and_timeout(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.return_value = []
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client, max_items=3, max_timeout_secs=60)

    tool._run(url='https://example.com', max_crawl_pages=100, timeout_secs=9999)

    call_kwargs = mock_tools_client.crawl_website.call_args
    assert call_kwargs.kwargs['max_crawl_pages'] == 3
    assert call_kwargs.kwargs['timeout_secs'] == 60


def test_web_crawler_tool_clamps_depth(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.return_value = []
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client, max_crawl_depth=2)

    tool._run(url='https://example.com', max_crawl_depth=999)
    assert mock_tools_client.crawl_website.call_args.kwargs['max_crawl_depth'] == 2

    mock_tools_client.crawl_website.reset_mock()
    tool._run(url='https://example.com', max_crawl_depth=-1)
    assert mock_tools_client.crawl_website.call_args.kwargs['max_crawl_depth'] == 0


def test_web_crawler_tool_empty_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.return_value = []
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client)

    result = tool._run(url='https://example.com')

    assert json.loads(result) == []


def test_web_crawler_tool_failure_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.crawl_website.side_effect = RuntimeError('Actor run run-bad ended with status TIMED-OUT.')
    tool = make_tool(ApifyWebCrawlerTool, mock_tools_client)

    with pytest.raises(ToolException, match='TIMED-OUT'):
        tool._run(url='https://example.com')


def test_web_crawler_tool_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyWebCrawlerTool()


# ---------------------------------------------------------------------------
# Metadata & inheritance
# ---------------------------------------------------------------------------


def test_actor_tools_inherit_from_generic_base() -> None:
    for tool_cls in (ApifyGoogleSearchTool, ApifyWebCrawlerTool):
        assert issubclass(tool_cls, _ApifyGenericTool), f'{tool_cls.__name__} must extend _ApifyGenericTool'


def test_actor_tools_have_correct_metadata() -> None:
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        tools = [
            ApifyGoogleSearchTool(apify_api_token=SecretStr('dummy')),
            ApifyWebCrawlerTool(apify_api_token=SecretStr('dummy')),
        ]

    expected_names = ['apify_google_search', 'apify_web_crawler']
    for tool, expected_name in zip(tools, expected_names):
        assert tool.name == expected_name
        assert tool.description
        assert tool.args_schema is not None
        assert tool.handle_tool_error is True


def test_apify_actor_tools_list() -> None:
    assert set(APIFY_ACTOR_TOOLS) == {ApifyGoogleSearchTool, ApifyWebCrawlerTool}
    assert len(APIFY_ACTOR_TOOLS) == 2


# ---------------------------------------------------------------------------
# US-4 Search & Crawling tools — happy paths
# ---------------------------------------------------------------------------


def test_rag_web_browser_tool_returns_json(mock_tools_client: MagicMock) -> None:
    items = [{'crawledUrl': 'https://example.com/1', 'text': 'hi'}]
    mock_tools_client.rag_web_browser_search.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyRAGWebBrowserTool, mock_tools_client)

    parsed = json.loads(tool._run(query='what is langchain', max_results=3))

    assert parsed['run']['run_id'] == SUCCEEDED_RUN['id']
    assert parsed['run']['status'] == 'SUCCEEDED'
    assert parsed['items'] == items
    mock_tools_client.rag_web_browser_search.assert_called_once_with(
        'what is langchain',
        max_results=3,
        timeout_secs=tool.max_timeout_secs,
    )


def test_google_maps_tool_returns_json(mock_tools_client: MagicMock) -> None:
    items = [{'name': 'Cafe A', 'address': 'Berlin'}]
    mock_tools_client.google_maps_search.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyGoogleMapsTool, mock_tools_client)

    parsed = json.loads(tool._run(query='cafe in Berlin', max_results=2, language='en'))

    assert parsed['run']['dataset_id'] == SUCCEEDED_RUN['defaultDatasetId']
    assert parsed['items'] == items
    mock_tools_client.google_maps_search.assert_called_once_with(
        'cafe in Berlin',
        max_results=2,
        language='en',
        timeout_secs=tool.max_timeout_secs,
    )


def test_youtube_tool_returns_json(mock_tools_client: MagicMock) -> None:
    items = [{'title': 'Vid 1'}]
    mock_tools_client.youtube_scrape.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyYouTubeScraperTool, mock_tools_client)

    parsed = json.loads(tool._run(search_query='langchain', search_type='search', max_results=4))

    assert parsed['items'] == items
    mock_tools_client.youtube_scrape.assert_called_once_with(
        search_query='langchain',
        search_type='search',
        max_results=4,
        timeout_secs=tool.max_timeout_secs,
    )


def test_youtube_tool_invalid_search_type_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.youtube_scrape.side_effect = ValueError('Invalid search_type playlist')
    tool = make_tool(ApifyYouTubeScraperTool, mock_tools_client)

    with pytest.raises(ToolException, match='Invalid search_type'):
        tool._run(search_query='x', search_type='search')


def test_ecommerce_tool_returns_json(mock_tools_client: MagicMock) -> None:
    items = [{'sku': 'A1', 'price': 9.99}]
    mock_tools_client.ecommerce_scrape.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyEcommerceScraperTool, mock_tools_client)

    parsed = json.loads(tool._run(url='https://shop.example.com/cat', max_results=5))

    assert parsed['items'] == items
    mock_tools_client.ecommerce_scrape.assert_called_once_with(
        'https://shop.example.com/cat',
        max_results=5,
        timeout_secs=tool.max_timeout_secs,
    )


# ---------------------------------------------------------------------------
# US-4 Search & Crawling tools — parametrized error / empty / handle_tool_error
# ---------------------------------------------------------------------------

# Each entry: (tool_class, helper_attribute_name, kwargs_for_run)
_TOOL_INVOCATIONS: list[tuple[type[_ApifyGenericTool], str, dict]] = [
    (ApifyRAGWebBrowserTool, 'rag_web_browser_search', {'query': 'q'}),
    (ApifyGoogleMapsTool, 'google_maps_search', {'query': 'q'}),
    (ApifyYouTubeScraperTool, 'youtube_scrape', {'search_query': 'q'}),
    (ApifyEcommerceScraperTool, 'ecommerce_scrape', {'url': 'https://example.com'}),
]


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _TOOL_INVOCATIONS)
def test_search_tool_runtime_error_raises_tool_exception(
    mock_tools_client: MagicMock,
    tool_cls: type[_ApifyGenericTool],
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    getattr(mock_tools_client, helper_attr).side_effect = RuntimeError(
        'Actor run run-bad ended with status FAILED.'
    )
    tool = make_tool(tool_cls, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(**run_kwargs)


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _TOOL_INVOCATIONS)
def test_search_tool_empty_dataset_returns_empty_items(
    mock_tools_client: MagicMock,
    tool_cls: type[_ApifyGenericTool],
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    getattr(mock_tools_client, helper_attr).return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(tool_cls, mock_tools_client)

    parsed = json.loads(tool._run(**run_kwargs))
    assert parsed['items'] == []
    assert parsed['run']['status'] == 'SUCCEEDED'


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _TOOL_INVOCATIONS)
def test_search_tool_handle_tool_error_swallows(
    mock_tools_client: MagicMock,
    tool_cls: type[_ApifyGenericTool],
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    """``handle_tool_error=True`` (inherited) means ``invoke`` returns the error string."""
    getattr(mock_tools_client, helper_attr).side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = make_tool(tool_cls, mock_tools_client)

    result = tool.invoke(run_kwargs)
    assert 'FAILED' in result


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _TOOL_INVOCATIONS)
def test_search_tool_missing_token(
    monkeypatch: pytest.MonkeyPatch,
    tool_cls: type[_ApifyGenericTool],
    helper_attr: str,  # noqa: ARG001
    run_kwargs: dict,  # noqa: ARG001
) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        tool_cls()


def test_search_tools_inherit_from_generic_base() -> None:
    for tool_cls, _, _ in _TOOL_INVOCATIONS:
        assert issubclass(tool_cls, _ApifyGenericTool), f'{tool_cls.__name__} must extend _ApifyGenericTool'


def test_search_tools_have_correct_metadata() -> None:
    expected_names = {
        ApifyRAGWebBrowserTool: 'apify_rag_web_browser',
        ApifyGoogleMapsTool: 'apify_google_maps',
        ApifyYouTubeScraperTool: 'apify_youtube_scraper',
        ApifyEcommerceScraperTool: 'apify_ecommerce_scraper',
    }
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        for tool_cls, expected_name in expected_names.items():
            tool = tool_cls(apify_api_token=SecretStr('dummy'))
            assert tool.name == expected_name
            assert tool.description
            assert tool.args_schema is not None
            assert tool.handle_tool_error is True


def test_apify_search_tools_list() -> None:
    assert set(APIFY_SEARCH_TOOLS) == {
        ApifyRAGWebBrowserTool,
        ApifyGoogleMapsTool,
        ApifyYouTubeScraperTool,
        ApifyEcommerceScraperTool,
    }
    assert len(APIFY_SEARCH_TOOLS) == 4
