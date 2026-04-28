from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from langchain_apify import APIFY_ACTOR_TOOLS, ApifyGoogleSearchTool, ApifyWebCrawlerTool
from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import _ApifyGenericTool
from tests.unit_tests.conftest import make_tool

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
