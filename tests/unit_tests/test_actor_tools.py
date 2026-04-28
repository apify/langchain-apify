from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_apify._actor_tools import (
    ApifyFacebookPostsScraperTool,
    ApifyInstagramScraperTool,
    ApifyLinkedInProfileDetailTool,
    ApifyLinkedInProfilePostsTool,
    ApifyLinkedInProfileSearchTool,
    ApifyTikTokScraperTool,
    ApifyTwitterScraperTool,
)
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool

EXPECTED_RUN_META: dict = {
    'run_id': 'run-abc',
    'status': 'SUCCEEDED',
    'dataset_id': 'dataset-xyz',
    'started_at': '2025-01-01T00:00:00.000Z',
    'finished_at': '2025-01-01T00:01:00.000Z',
}


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
        timeout_secs=600,
    )


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
        timeout_secs=600,
    )


def test_facebook_tool_passes_only_posts_newer_than(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_posts_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyFacebookPostsScraperTool, mock_tools_client)

    tool._run(page_url='https://www.facebook.com/humansofnewyork/', only_posts_newer_than='2025-01-01')

    assert mock_tools_client.facebook_posts_scrape.call_args.kwargs['only_posts_newer_than'] == '2025-01-01'


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
_TOOL_INVOCATIONS: list[tuple[type, str, dict]] = [
    (ApifyInstagramScraperTool, 'instagram_scrape', {'search_type': 'user', 'search_query': 'apify'}),
    (ApifyLinkedInProfilePostsTool, 'linkedin_profile_posts', {'profile_url': 'satyanadella'}),
    (ApifyLinkedInProfileSearchTool, 'linkedin_profile_search', {'query': 'Founder'}),
    (ApifyLinkedInProfileDetailTool, 'linkedin_profile_detail', {'profile_url': 'neal-mohan'}),
    (ApifyTwitterScraperTool, 'twitter_scrape', {'search_query': 'apify'}),
    (ApifyTikTokScraperTool, 'tiktok_scrape', {'search_query': 'cooking'}),
    (ApifyFacebookPostsScraperTool, 'facebook_posts_scrape', {'page_url': 'https://www.facebook.com/x/'}),
]


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TOOL_INVOCATIONS)
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
