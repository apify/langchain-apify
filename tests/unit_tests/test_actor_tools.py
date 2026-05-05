from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from langchain_apify import (
    APIFY_SEARCH_TOOLS,
    ApifyEcommerceScraperTool,
    ApifyFacebookPostsScraperTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyInstagramScraperTool,
    ApifyLinkedInProfileDetailTool,
    ApifyLinkedInProfilePostsTool,
    ApifyLinkedInProfileSearchTool,
    ApifyRAGWebBrowserTool,
    ApifyTikTokScraperTool,
    ApifyTwitterScraperTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
)
from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import _ApifyGenericTool
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool

EXPECTED_RUN_META: dict = {
    'run_id': 'run-abc',
    'status': 'SUCCEEDED',
    'dataset_id': 'dataset-xyz',
    'started_at': '2025-01-01T00:00:00.000Z',
    'finished_at': '2025-01-01T00:01:00.000Z',
}

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


# ---------------------------------------------------------------------------
# Missing token (shared base behavior)
# ---------------------------------------------------------------------------


def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyInstagramScraperTool()


# ---------------------------------------------------------------------------
# ApifyInstagramScraperTool
# ---------------------------------------------------------------------------


def test_instagram_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.instagram_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyInstagramScraperTool, mock_tools_client)

    result = tool._run(search_type='user', search_query='apify', max_results=10)

    parsed = json.loads(result)
    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS


def test_instagram_tool_passes_params(mock_tools_client: MagicMock) -> None:
    mock_tools_client.instagram_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyInstagramScraperTool, mock_tools_client)

    tool._run(
        search_type='hashtag',
        search_query='#travel',
        max_results=5,
        only_posts_newer_than='1 week',
    )

    mock_tools_client.instagram_scrape.assert_called_once_with(
        search_type='hashtag',
        search_query='#travel',
        max_results=5,
        only_posts_newer_than='1 week',
        timeout_secs=600,
    )


def test_instagram_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.instagram_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyInstagramScraperTool, mock_tools_client, max_items=3)

    tool._run(search_type='user', search_query='apify', max_results=100)

    assert mock_tools_client.instagram_scrape.call_args.kwargs['max_results'] == 3


def test_instagram_tool_runtime_error_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.instagram_scrape.side_effect = RuntimeError('Actor run run-X ended with status FAILED.')
    tool = make_tool(ApifyInstagramScraperTool, mock_tools_client)

    with pytest.raises(ToolException, match='run-X'):
        tool._run(search_type='user', search_query='apify')


# ---------------------------------------------------------------------------
# ApifyLinkedInProfilePostsTool
# ---------------------------------------------------------------------------


def test_linkedin_posts_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_posts.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyLinkedInProfilePostsTool, mock_tools_client)

    result = tool._run(profile_url='satyanadella', max_results=10)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.linkedin_profile_posts.assert_called_once_with(
        profile_url='satyanadella',
        max_results=10,
        timeout_secs=600,
    )


def test_linkedin_posts_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_posts.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyLinkedInProfilePostsTool, mock_tools_client, max_items=5)

    tool._run(profile_url='satyanadella', max_results=999)

    assert mock_tools_client.linkedin_profile_posts.call_args.kwargs['max_results'] == 5


# ---------------------------------------------------------------------------
# ApifyLinkedInProfileSearchTool
# ---------------------------------------------------------------------------


def test_linkedin_search_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_search.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyLinkedInProfileSearchTool, mock_tools_client)

    result = tool._run(query='Founder', max_results=10)
    parsed = json.loads(result)

    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.linkedin_profile_search.assert_called_once_with(
        query='Founder',
        max_results=10,
        timeout_secs=600,
    )


def test_linkedin_search_tool_default_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_search.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyLinkedInProfileSearchTool, mock_tools_client)

    tool._run(query='CTO')

    assert mock_tools_client.linkedin_profile_search.call_args.kwargs['max_results'] == 10


# ---------------------------------------------------------------------------
# ApifyLinkedInProfileDetailTool
# ---------------------------------------------------------------------------


def test_linkedin_detail_tool_happy_path(mock_tools_client: MagicMock) -> None:
    profile_item = [{'firstName': 'Neal', 'lastName': 'Mohan'}]
    mock_tools_client.linkedin_profile_detail.return_value = (SUCCEEDED_RUN, profile_item)
    tool = make_tool(ApifyLinkedInProfileDetailTool, mock_tools_client)

    result = tool._run(profile_url='neal-mohan', include_email=True)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == profile_item
    mock_tools_client.linkedin_profile_detail.assert_called_once_with(
        profile_url='neal-mohan',
        include_email=True,
        timeout_secs=600,
    )


def test_linkedin_detail_tool_default_include_email_false(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_detail.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyLinkedInProfileDetailTool, mock_tools_client)

    tool._run(profile_url='neal-mohan')

    assert mock_tools_client.linkedin_profile_detail.call_args.kwargs['include_email'] is False


# ---------------------------------------------------------------------------
# ApifyTwitterScraperTool
# ---------------------------------------------------------------------------


def test_twitter_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.twitter_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyTwitterScraperTool, mock_tools_client)

    result = tool._run(search_query='apify', max_results=20)
    parsed = json.loads(result)

    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.twitter_scrape.assert_called_once_with(
        search_query='apify',
        search_mode='search',
        max_results=20,
        start=None,
        end=None,
        sort=None,
        timeout_secs=600,
    )


def test_twitter_tool_passes_sort(mock_tools_client: MagicMock) -> None:
    mock_tools_client.twitter_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyTwitterScraperTool, mock_tools_client)

    tool._run(search_query='apify', sort='Top')

    kwargs = mock_tools_client.twitter_scrape.call_args.kwargs
    assert kwargs['sort'] == 'Top'


def test_twitter_tool_passes_date_range(mock_tools_client: MagicMock) -> None:
    mock_tools_client.twitter_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyTwitterScraperTool, mock_tools_client)

    tool._run(search_query='apify', search_mode='user', start='2025-01-01', end='2025-02-01')

    kwargs = mock_tools_client.twitter_scrape.call_args.kwargs
    assert kwargs['search_mode'] == 'user'
    assert kwargs['start'] == '2025-01-01'
    assert kwargs['end'] == '2025-02-01'


def test_twitter_tool_value_error_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.twitter_scrape.side_effect = ValueError('Unsupported Twitter search_mode')
    tool = make_tool(ApifyTwitterScraperTool, mock_tools_client)

    with pytest.raises(ToolException, match='Unsupported Twitter search_mode'):
        tool._run(search_query='apify', search_mode='replies')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ApifyTikTokScraperTool
# ---------------------------------------------------------------------------


def test_tiktok_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.tiktok_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyTikTokScraperTool, mock_tools_client)

    result = tool._run(search_query='cooking', search_type='search', max_results=12)
    parsed = json.loads(result)

    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.tiktok_scrape.assert_called_once_with(
        search_query='cooking',
        search_type='search',
        max_results=12,
        timeout_secs=600,
    )


def test_tiktok_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.tiktok_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyTikTokScraperTool, mock_tools_client, max_items=4)

    tool._run(search_query='cooking', max_results=500)

    assert mock_tools_client.tiktok_scrape.call_args.kwargs['max_results'] == 4


def test_tiktok_tool_passes_post_search_type(mock_tools_client: MagicMock) -> None:
    mock_tools_client.tiktok_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyTikTokScraperTool, mock_tools_client)

    tool._run(search_query='https://www.tiktok.com/@charlidamelio/video/123', search_type='post')

    assert mock_tools_client.tiktok_scrape.call_args.kwargs['search_type'] == 'post'


# ---------------------------------------------------------------------------
# ApifyFacebookPostsScraperTool
# ---------------------------------------------------------------------------


def test_facebook_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_posts_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyFacebookPostsScraperTool, mock_tools_client)

    result = tool._run(page_url='https://www.facebook.com/humansofnewyork/', max_results=15)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.facebook_posts_scrape.assert_called_once_with(
        page_url='https://www.facebook.com/humansofnewyork/',
        max_results=15,
        only_posts_newer_than=None,
        only_posts_older_than=None,
        timeout_secs=600,
    )


def test_facebook_tool_passes_only_posts_newer_than(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_posts_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyFacebookPostsScraperTool, mock_tools_client)

    tool._run(page_url='https://www.facebook.com/humansofnewyork/', only_posts_newer_than='2025-01-01')

    assert mock_tools_client.facebook_posts_scrape.call_args.kwargs['only_posts_newer_than'] == '2025-01-01'


def test_facebook_tool_passes_only_posts_older_than(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_posts_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyFacebookPostsScraperTool, mock_tools_client)

    tool._run(page_url='https://www.facebook.com/humansofnewyork/', only_posts_older_than='2025-12-31')

    assert mock_tools_client.facebook_posts_scrape.call_args.kwargs['only_posts_older_than'] == '2025-12-31'


def test_facebook_tool_runtime_error_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_posts_scrape.side_effect = RuntimeError('Network error')
    tool = make_tool(ApifyFacebookPostsScraperTool, mock_tools_client)

    with pytest.raises(ToolException, match='Network error'):
        tool._run(page_url='https://www.facebook.com/humansofnewyork/')


# ---------------------------------------------------------------------------
# Empty results - tools should still return valid JSON
# ---------------------------------------------------------------------------


def test_tool_returns_valid_json_for_empty_items(mock_tools_client: MagicMock) -> None:
    mock_tools_client.linkedin_profile_search.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyLinkedInProfileSearchTool, mock_tools_client)

    result = tool._run(query='nonexistent')
    parsed = json.loads(result)

    assert parsed['items'] == []
    assert parsed['run']['status'] == 'SUCCEEDED'


# ---------------------------------------------------------------------------
# Search & crawling Actor tools — happy paths
# ---------------------------------------------------------------------------


def test_rag_web_browser_tool_returns_json(mock_tools_client: MagicMock) -> None:
    items = [
        {
            'crawledUrl': 'https://example.com/1',
            'metadata': {'url': 'https://example.com/1', 'title': 'Page 1'},
            'markdown': '# Page 1',
            'text': 'Page 1 plain',
        },
        {
            'crawledUrl': 'https://example.com/2',
            'metadata': {'title': 'Page 2'},
            'text': 'Page 2 plain',
        },
    ]
    mock_tools_client.rag_web_browser_search.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyRAGWebBrowserTool, mock_tools_client)

    parsed = json.loads(tool._run(query='what is langchain', max_results=3))

    assert parsed == [
        {'url': 'https://example.com/1', 'title': 'Page 1', 'content': '# Page 1'},
        {'url': 'https://example.com/2', 'title': 'Page 2', 'content': 'Page 2 plain'},
    ]
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

    parsed = json.loads(tool._run(url='https://shop.example.com/p/123', max_results=5))

    assert parsed['items'] == items
    mock_tools_client.ecommerce_scrape.assert_called_once_with(
        'https://shop.example.com/p/123',
        url_type='product',
        max_results=5,
        timeout_secs=tool.max_timeout_secs,
    )


def test_ecommerce_tool_category_mode_passes_url_type(mock_tools_client: MagicMock) -> None:
    items = [{'sku': 'B2', 'price': 19.99}]
    mock_tools_client.ecommerce_scrape.return_value = (SUCCEEDED_RUN, items)
    tool = make_tool(ApifyEcommerceScraperTool, mock_tools_client)

    parsed = json.loads(tool._run(url='https://shop.example.com/cat/42', url_type='category', max_results=8))

    assert parsed['items'] == items
    mock_tools_client.ecommerce_scrape.assert_called_once_with(
        'https://shop.example.com/cat/42',
        url_type='category',
        max_results=8,
        timeout_secs=tool.max_timeout_secs,
    )


def test_ecommerce_tool_invalid_url_type_raises_tool_exception(mock_tools_client: MagicMock) -> None:
    mock_tools_client.ecommerce_scrape.side_effect = ValueError('Invalid url_type listing')
    tool = make_tool(ApifyEcommerceScraperTool, mock_tools_client)

    with pytest.raises(ToolException, match='Invalid url_type'):
        tool._run(url='https://shop.example.com', url_type='product')


# ---------------------------------------------------------------------------
# Search & crawling Actor tools — parametrized error / empty
# ---------------------------------------------------------------------------

# Each entry: (tool_class, helper_attribute_name, kwargs_for_run)
_SEARCH_TOOL_INVOCATIONS: list[tuple[type[_ApifyGenericTool], str, dict]] = [
    (ApifyRAGWebBrowserTool, 'rag_web_browser_search', {'query': 'q'}),
    (ApifyGoogleMapsTool, 'google_maps_search', {'query': 'q'}),
    (ApifyYouTubeScraperTool, 'youtube_scrape', {'search_query': 'q'}),
    (ApifyEcommerceScraperTool, 'ecommerce_scrape', {'url': 'https://example.com'}),
]

# Tools that return the {run, items} envelope on success.
_SEARCH_ENVELOPE_TOOL_INVOCATIONS: list[tuple[type[_ApifyGenericTool], str, dict]] = [
    (ApifyGoogleMapsTool, 'google_maps_search', {'query': 'q'}),
    (ApifyYouTubeScraperTool, 'youtube_scrape', {'search_query': 'q'}),
    (ApifyEcommerceScraperTool, 'ecommerce_scrape', {'url': 'https://example.com'}),
]


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _SEARCH_TOOL_INVOCATIONS)
def test_search_tool_runtime_error_raises_tool_exception(
    mock_tools_client: MagicMock,
    tool_cls: type,
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    getattr(mock_tools_client, helper_attr).side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = make_tool(tool_cls, mock_tools_client)

    with pytest.raises(ToolException, match='FAILED'):
        tool._run(**run_kwargs)


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _SEARCH_ENVELOPE_TOOL_INVOCATIONS)
def test_search_tool_empty_dataset_returns_empty_items(
    mock_tools_client: MagicMock,
    tool_cls: type,
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    getattr(mock_tools_client, helper_attr).return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(tool_cls, mock_tools_client)

    parsed = json.loads(tool._run(**run_kwargs))
    assert parsed['items'] == []
    assert parsed['run']['status'] == 'SUCCEEDED'


def test_rag_web_browser_tool_empty_dataset_returns_empty_array(mock_tools_client: MagicMock) -> None:
    mock_tools_client.rag_web_browser_search.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyRAGWebBrowserTool, mock_tools_client)

    assert json.loads(tool._run(query='q')) == []


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _SEARCH_TOOL_INVOCATIONS)
def test_search_tool_handle_tool_error_swallows(
    mock_tools_client: MagicMock,
    tool_cls: type,
    helper_attr: str,
    run_kwargs: dict,
) -> None:
    """``handle_tool_error=True`` (inherited) means ``invoke`` returns the error string."""
    getattr(mock_tools_client, helper_attr).side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    tool = make_tool(tool_cls, mock_tools_client)

    result = tool.invoke(run_kwargs)
    assert 'FAILED' in result


@pytest.mark.parametrize(('tool_cls', 'helper_attr', 'run_kwargs'), _SEARCH_TOOL_INVOCATIONS)
def test_search_tool_missing_token(
    monkeypatch: pytest.MonkeyPatch,
    tool_cls: type,
    helper_attr: str,  # noqa: ARG001
    run_kwargs: dict,  # noqa: ARG001
) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        tool_cls()


def test_search_tools_inherit_from_generic_base() -> None:
    for tool_cls, _, _ in _SEARCH_TOOL_INVOCATIONS:
        assert issubclass(tool_cls, _ApifyGenericTool), f'{tool_cls.__name__} must extend _ApifyGenericTool'


def test_search_tools_have_correct_metadata() -> None:
    cases: list[tuple[type, str]] = [
        (ApifyRAGWebBrowserTool, 'apify_rag_web_browser'),
        (ApifyGoogleMapsTool, 'apify_google_maps'),
        (ApifyYouTubeScraperTool, 'apify_youtube_scraper'),
        (ApifyEcommerceScraperTool, 'apify_ecommerce_scraper'),
    ]
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        for tool_cls, expected_name in cases:
            tool = tool_cls(apify_api_token=SecretStr('dummy'))
            assert tool.name == expected_name
            assert tool.description
            assert tool.args_schema is not None
            assert tool.handle_tool_error is True


def test_apify_search_tools_list() -> None:
    assert set(APIFY_SEARCH_TOOLS) == {
        ApifyGoogleSearchTool,
        ApifyWebCrawlerTool,
        ApifyRAGWebBrowserTool,
        ApifyGoogleMapsTool,
        ApifyYouTubeScraperTool,
        ApifyEcommerceScraperTool,
    }
    assert len(APIFY_SEARCH_TOOLS) == 6


# ---------------------------------------------------------------------------
# handle_tool_error is True on every social tool (existing base behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'tool_cls',
    [
        ApifyInstagramScraperTool,
        ApifyLinkedInProfilePostsTool,
        ApifyLinkedInProfileSearchTool,
        ApifyLinkedInProfileDetailTool,
        ApifyTwitterScraperTool,
        ApifyTikTokScraperTool,
        ApifyFacebookPostsScraperTool,
    ],
)
def test_social_tool_handle_tool_error_enabled(tool_cls: type, mock_tools_client: MagicMock) -> None:
    tool = make_tool(tool_cls, mock_tools_client)
    assert tool.handle_tool_error is True


# ---------------------------------------------------------------------------
# Per-tool RuntimeError -> ToolException coverage
# ---------------------------------------------------------------------------

# (tool_cls, client_method_name, _run kwargs)
_SOCIAL_TOOL_INVOCATIONS: list[tuple[type, str, dict]] = [
    (ApifyInstagramScraperTool, 'instagram_scrape', {'search_type': 'user', 'search_query': 'apify'}),
    (ApifyLinkedInProfilePostsTool, 'linkedin_profile_posts', {'profile_url': 'satyanadella'}),
    (ApifyLinkedInProfileSearchTool, 'linkedin_profile_search', {'query': 'Founder'}),
    (ApifyLinkedInProfileDetailTool, 'linkedin_profile_detail', {'profile_url': 'neal-mohan'}),
    (ApifyTwitterScraperTool, 'twitter_scrape', {'search_query': 'apify'}),
    (ApifyTikTokScraperTool, 'tiktok_scrape', {'search_query': 'cooking'}),
    (ApifyFacebookPostsScraperTool, 'facebook_posts_scrape', {'page_url': 'https://www.facebook.com/x/'}),
]


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _SOCIAL_TOOL_INVOCATIONS)
def test_social_tool_runtime_error_raises_tool_exception(
    tool_cls: type,
    method_name: str,
    run_kwargs: dict,
    mock_tools_client: MagicMock,
) -> None:
    getattr(mock_tools_client, method_name).side_effect = RuntimeError(
        'Actor run run-XYZ ended with status FAILED.',
    )
    tool = make_tool(tool_cls, mock_tools_client)

    with pytest.raises(ToolException, match='run-XYZ'):
        tool._run(**run_kwargs)


# ---------------------------------------------------------------------------
# Per-tool empty-dataset coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _SOCIAL_TOOL_INVOCATIONS)
def test_social_tool_returns_valid_json_for_empty_items(
    tool_cls: type,
    method_name: str,
    run_kwargs: dict,
    mock_tools_client: MagicMock,
) -> None:
    getattr(mock_tools_client, method_name).return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(tool_cls, mock_tools_client)

    result = tool._run(**run_kwargs)
    parsed = json.loads(result)

    assert parsed['items'] == []
    assert parsed['run'] == EXPECTED_RUN_META
