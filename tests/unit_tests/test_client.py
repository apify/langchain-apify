from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from langchain_apify._client import ApifyToolsClient
from tests.unit_tests.conftest import FAILED_RUN, SAMPLE_ITEMS, SUCCEEDED_RUN

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_with_explicit_token(mock_apify_client: MagicMock) -> None:
    with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client) as mock_create:
        c = ApifyToolsClient(apify_api_token='my-token')
        mock_create.assert_called_once()
        assert c._client is mock_apify_client


def test_init_with_env_token(monkeypatch: pytest.MonkeyPatch, mock_apify_client: MagicMock) -> None:
    monkeypatch.setenv('APIFY_API_TOKEN', 'env-token')
    with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client):
        c = ApifyToolsClient()
        assert c._client is mock_apify_client


def test_init_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyToolsClient()


# ---------------------------------------------------------------------------
# run_actor
# ---------------------------------------------------------------------------


def test_run_actor_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN

    result = client.run_actor('apify/test-actor', run_input={'key': 'val'})

    mock_apify_client.actor.assert_called_once_with('apify/test-actor')
    mock_apify_client.actor.return_value.call.assert_called_once_with(
        run_input={'key': 'val'}, timeout_secs=300, logger=None
    )
    assert result == SUCCEEDED_RUN


def test_run_actor_with_memory(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN

    client.run_actor('apify/test-actor', memory_mbytes=512)

    mock_apify_client.actor.return_value.call.assert_called_once_with(
        run_input=None, timeout_secs=300, logger=None, memory_mbytes=512
    )


def test_run_actor_failed_status_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        client.run_actor('apify/test-actor')


# ---------------------------------------------------------------------------
# get_dataset_items
# ---------------------------------------------------------------------------


def test_get_dataset_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    items = client.get_dataset_items('dataset-xyz', limit=50, offset=10)

    mock_apify_client.dataset.assert_called_once_with('dataset-xyz')
    mock_apify_client.dataset.return_value.list_items.assert_called_once_with(limit=50, offset=10, clean=True)
    assert items == SAMPLE_ITEMS


def test_get_dataset_items_empty(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.return_value.items = []

    items = client.get_dataset_items('dataset-empty')
    assert items == []


# ---------------------------------------------------------------------------
# run_actor_and_get_items
# ---------------------------------------------------------------------------


def test_run_actor_and_get_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    run, items = client.run_actor_and_get_items('apify/test-actor', run_input={'q': '1'})

    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS
    mock_apify_client.dataset.assert_called_once_with('dataset-xyz')


def test_run_actor_and_get_items_missing_dataset_id_raises(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    run_no_dataset = {**SUCCEEDED_RUN, 'defaultDatasetId': None}
    mock_apify_client.actor.return_value.call.return_value = run_no_dataset

    with pytest.raises(RuntimeError, match='no default dataset ID'):
        client.run_actor_and_get_items('apify/test-actor')


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------


def test_run_task_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN

    result = client.run_task('user/my-task', task_input={'key': 'val'})

    mock_apify_client.task.assert_called_once_with('user/my-task')
    mock_apify_client.task.return_value.call.assert_called_once_with(task_input={'key': 'val'}, timeout_secs=300)
    assert result == SUCCEEDED_RUN


def test_run_task_failed_status_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        client.run_task('user/my-task')


# ---------------------------------------------------------------------------
# run_task_and_get_items
# ---------------------------------------------------------------------------


def test_run_task_and_get_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    run, items = client.run_task_and_get_items('user/my-task')

    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_run_task_and_get_items_missing_dataset_id_raises(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    run_no_dataset = {**SUCCEEDED_RUN, 'defaultDatasetId': None}
    mock_apify_client.task.return_value.call.return_value = run_no_dataset

    with pytest.raises(RuntimeError, match='no default dataset ID'):
        client.run_task_and_get_items('user/my-task')


# ---------------------------------------------------------------------------
# scrape_url
# ---------------------------------------------------------------------------


def test_scrape_url_returns_markdown(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'markdown': '# Hello', 'text': 'Hello', 'url': 'https://example.com'},
    ]

    content = client.scrape_url('https://example.com')
    assert content == '# Hello'


def test_scrape_url_falls_back_to_text(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'text': 'Plain text content', 'url': 'https://example.com'},
    ]

    content = client.scrape_url('https://example.com')
    assert content == 'Plain text content'


def test_scrape_url_empty_items_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = []

    with pytest.raises(RuntimeError, match='No content extracted'):
        client.scrape_url('https://example.com')


def test_scrape_url_empty_content_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'markdown': '', 'text': '', 'url': 'https://example.com'},
    ]

    with pytest.raises(RuntimeError, match='No content extracted'):
        client.scrape_url('https://example.com')


# ---------------------------------------------------------------------------
# _check_run_status
# ---------------------------------------------------------------------------


def test_check_run_status_succeeded() -> None:
    ApifyToolsClient._check_run_status({'id': 'run-ok', 'status': 'SUCCEEDED'})


def test_check_run_status_failed() -> None:
    with pytest.raises(RuntimeError, match='run-bad'):
        ApifyToolsClient._check_run_status({'id': 'run-bad', 'status': 'FAILED'})


def test_check_run_status_failed_includes_status_message() -> None:
    with pytest.raises(RuntimeError, match='Actor exited out of memory'):
        ApifyToolsClient._check_run_status(
            {'id': 'run-oom', 'status': 'FAILED', 'statusMessage': 'Actor exited out of memory'},
        )


# ---------------------------------------------------------------------------
# None returns from actor/task .call()
# ---------------------------------------------------------------------------


def test_run_actor_none_return_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match='returned no run details'):
        client.run_actor('apify/broken-actor')


def test_run_task_none_return_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match='returned no run details'):
        client.run_task('user/broken-task')


# ---------------------------------------------------------------------------
# Transport-error wrapping (httpx / ApifyClientError -> RuntimeError)
# ---------------------------------------------------------------------------


def test_run_actor_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.side_effect = httpx.ConnectError('conn refused')

    with pytest.raises(RuntimeError, match='Apify Actor call failed'):
        client.run_actor('apify/test-actor')


def test_get_dataset_items_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('timeout')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.get_dataset_items('dataset-xyz')


def test_run_actor_and_get_items_dataset_fetch_network_error(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('reset')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.run_actor_and_get_items('apify/test-actor')


def test_run_task_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.side_effect = httpx.ConnectError('conn refused')

    with pytest.raises(RuntimeError, match='Apify task call failed'):
        client.run_task('user/my-task')


def test_run_task_and_get_items_dataset_fetch_network_error(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('reset')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.run_task_and_get_items('user/my-task')


def test_run_actor_programming_error_propagates(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    """Non-transport exceptions (programming errors) must NOT be wrapped as RuntimeError."""
    mock_apify_client.actor.return_value.call.side_effect = AttributeError('bug in SDK')

    with pytest.raises(AttributeError, match='bug in SDK'):
        client.run_actor('apify/test-actor')


# ---------------------------------------------------------------------------
# instagram_scrape
# ---------------------------------------------------------------------------


def _setup_run_and_items(mock_apify_client: MagicMock, items: list[dict] | None = None) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = items or SAMPLE_ITEMS


def test_instagram_scrape_user_builds_profile_url(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.instagram_scrape('user', 'apify', max_results=5)

    mock_apify_client.actor.assert_called_once_with('apify/instagram-scraper')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {
        'directUrls': ['https://www.instagram.com/apify/'],
        'resultsType': 'posts',
        'resultsLimit': 5,
    }
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_instagram_scrape_hashtag_builds_tag_url(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.instagram_scrape('hashtag', '#travel', max_results=10)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['directUrls'] == ['https://www.instagram.com/explore/tags/travel/']
    assert call_kwargs['run_input']['resultsType'] == 'posts'


def test_instagram_scrape_comments_uses_comments_results_type(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    _setup_run_and_items(mock_apify_client)

    client.instagram_scrape('comments', 'https://www.instagram.com/p/ABC123/', max_results=15)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['resultsType'] == 'comments'
    assert call_kwargs['run_input']['directUrls'] == ['https://www.instagram.com/p/ABC123/']


def test_instagram_scrape_passes_only_posts_newer_than(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.instagram_scrape('user', 'apify', only_posts_newer_than='1 week')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['onlyPostsNewerThan'] == '1 week'


def test_instagram_scrape_invalid_search_type_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='Unsupported Instagram search_type'):
        client.instagram_scrape('reels', 'apify')


# ---------------------------------------------------------------------------
# linkedin_profile_posts
# ---------------------------------------------------------------------------


def test_linkedin_profile_posts_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.linkedin_profile_posts('https://www.linkedin.com/in/satyanadella', max_results=30)

    mock_apify_client.actor.assert_called_once_with('apimaestro/linkedin-profile-posts')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {
        'username': 'https://www.linkedin.com/in/satyanadella',
        'total_posts': 30,
    }
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


# ---------------------------------------------------------------------------
# linkedin_profile_search
# ---------------------------------------------------------------------------


def test_linkedin_profile_search_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.linkedin_profile_search('Founder', max_results=25)

    mock_apify_client.actor.assert_called_once_with('harvestapi/linkedin-profile-search')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'searchQuery': 'Founder', 'maxItems': 25}


def test_linkedin_profile_search_default_max_results(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.linkedin_profile_search('CTO')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['maxItems'] == 10


# ---------------------------------------------------------------------------
# linkedin_profile_detail
# ---------------------------------------------------------------------------


def test_linkedin_profile_detail_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client, items=[{'firstName': 'Neal'}])

    run, items = client.linkedin_profile_detail('neal-mohan', include_email=True)

    mock_apify_client.actor.assert_called_once_with('apimaestro/linkedin-profile-detail')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'username': 'neal-mohan', 'includeEmail': True}
    assert run == SUCCEEDED_RUN
    assert items == [{'firstName': 'Neal'}]


def test_linkedin_profile_detail_default_include_email_false(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    _setup_run_and_items(mock_apify_client)

    client.linkedin_profile_detail('neal-mohan')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['includeEmail'] is False


# ---------------------------------------------------------------------------
# twitter_scrape
# ---------------------------------------------------------------------------


def test_twitter_scrape_search_mode(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.twitter_scrape('apify', max_results=50)

    mock_apify_client.actor.assert_called_once_with('apidojo/twitter-scraper-lite')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'maxItems': 50, 'searchTerms': ['apify']}


def test_twitter_scrape_user_mode_strips_at(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.twitter_scrape('@apify', search_mode='user', max_results=10)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'maxItems': 10, 'twitterHandles': ['apify']}


def test_twitter_scrape_replies_mode_uses_start_urls(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.twitter_scrape('https://x.com/apify/status/123', search_mode='replies')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['startUrls'] == ['https://x.com/apify/status/123']


def test_twitter_scrape_passes_date_range(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.twitter_scrape('apify', start='2025-01-01', end='2025-02-01')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['start'] == '2025-01-01'
    assert call_kwargs['run_input']['end'] == '2025-02-01'


def test_twitter_scrape_passes_sort(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.twitter_scrape('apify', sort='Top')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['sort'] == 'Top'


def test_twitter_scrape_invalid_mode_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='Unsupported Twitter search_mode'):
        client.twitter_scrape('apify', search_mode='followers')


# ---------------------------------------------------------------------------
# tiktok_scrape
# ---------------------------------------------------------------------------


def test_tiktok_scrape_search_mode(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.tiktok_scrape('cooking', max_results=12)

    mock_apify_client.actor.assert_called_once_with('clockworks/tiktok-scraper')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'resultsPerPage': 12, 'searchQueries': ['cooking']}


def test_tiktok_scrape_user_mode_strips_at(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.tiktok_scrape('@charlidamelio', search_type='user')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['profiles'] == ['charlidamelio']


def test_tiktok_scrape_hashtag_mode_strips_hash(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.tiktok_scrape('#fyp', search_type='hashtag')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['hashtags'] == ['fyp']


def test_tiktok_scrape_post_mode_uses_post_urls(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.tiktok_scrape('https://www.tiktok.com/@charlidamelio/video/123', search_type='post')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['postURLs'] == ['https://www.tiktok.com/@charlidamelio/video/123']


def test_tiktok_scrape_invalid_type_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='Unsupported TikTok search_type'):
        client.tiktok_scrape('cooking', search_type='trending')


# ---------------------------------------------------------------------------
# facebook_posts_scrape
# ---------------------------------------------------------------------------


def test_facebook_posts_scrape_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.facebook_posts_scrape('https://www.facebook.com/humansofnewyork/', max_results=15)

    mock_apify_client.actor.assert_called_once_with('apify/facebook-posts-scraper')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {
        'startUrls': [{'url': 'https://www.facebook.com/humansofnewyork/'}],
        'resultsLimit': 15,
    }
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_facebook_posts_scrape_passes_only_posts_newer_than(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    _setup_run_and_items(mock_apify_client)

    client.facebook_posts_scrape('https://www.facebook.com/humansofnewyork/', only_posts_newer_than='2025-01-01')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['onlyPostsNewerThan'] == '2025-01-01'


def test_facebook_posts_scrape_passes_only_posts_older_than(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    _setup_run_and_items(mock_apify_client)

    client.facebook_posts_scrape('https://www.facebook.com/humansofnewyork/', only_posts_older_than='2025-12-31')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['onlyPostsOlderThan'] == '2025-12-31'


# ---------------------------------------------------------------------------
# Failed run propagates from social helpers
# ---------------------------------------------------------------------------


def test_social_helper_propagates_failed_run(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        client.instagram_scrape('user', 'apify')


# ---------------------------------------------------------------------------
# _build_instagram_url
# ---------------------------------------------------------------------------


def test_build_instagram_url_passthrough_for_full_url() -> None:
    assert (
        ApifyToolsClient._build_instagram_url('post', 'https://www.instagram.com/p/abc/')
        == 'https://www.instagram.com/p/abc/'
    )


def test_build_instagram_url_user() -> None:
    assert ApifyToolsClient._build_instagram_url('user', '@apify') == 'https://www.instagram.com/apify/'


def test_build_instagram_url_hashtag() -> None:
    assert (
        ApifyToolsClient._build_instagram_url('hashtag', '#travel') == 'https://www.instagram.com/explore/tags/travel/'
    )


def test_build_instagram_url_post_from_id() -> None:
    assert ApifyToolsClient._build_instagram_url('post', 'ABC123') == 'https://www.instagram.com/p/ABC123/'
