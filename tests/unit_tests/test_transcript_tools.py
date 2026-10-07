from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_apify import (
    APIFY_TRANSCRIPT_TOOLS,
    ApifyFacebookAdsTranscriptTool,
    ApifyMediaTranscriberTool,
    ApifyYouTubeTranscriptTool,
)
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_tool

if TYPE_CHECKING:
    from langchain_apify._client import ApifyToolsClient

EXPECTED_RUN_META: dict = {
    'run_id': 'run-abc',
    'status': 'SUCCEEDED',
    'dataset_id': 'dataset-xyz',
    'started_at': '2025-01-01T00:00:00.000Z',
    'finished_at': '2025-01-01T00:01:00.000Z',
}


def _setup_run_and_items(mock_apify_client: MagicMock, items: list[dict] | None = None) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = items or SAMPLE_ITEMS


# ---------------------------------------------------------------------------
# Client - facebook_ads_transcript_scrape
# ---------------------------------------------------------------------------


def test_facebook_ads_transcript_scrape_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.facebook_ads_transcript_scrape(['fitness app'], max_results=12)

    mock_apify_client.actor.assert_called_once_with('steadyfetch/facebook-ads-transcript-scraper')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {
        'searchQueries': ['fitness app'],
        'country': 'US',
        'searchMaxAds': 12,
        'maxAds': 12,
    }
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_facebook_ads_transcript_scrape_passes_country(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    client.facebook_ads_transcript_scrape(['fitness app'], country='GB')

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input']['country'] == 'GB'


def test_facebook_ads_transcript_scrape_bounds_dataset_items(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    _setup_run_and_items(mock_apify_client)

    client.facebook_ads_transcript_scrape(['fitness app'], max_results=7)

    assert mock_apify_client.dataset.return_value.list_items.call_args.kwargs['limit'] == 7


def test_facebook_ads_transcript_scrape_empty_queries_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='search_queries'):
        client.facebook_ads_transcript_scrape([])


# ---------------------------------------------------------------------------
# Client - youtube_transcript_scrape
# ---------------------------------------------------------------------------


def test_youtube_transcript_scrape_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.youtube_transcript_scrape(
        ['https://www.youtube.com/watch?v=jNQXAC9IVRw'],
        max_results=5,
    )

    mock_apify_client.actor.assert_called_once_with('steadyfetch/youtube-transcript-scraper')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {
        'videoUrls': ['https://www.youtube.com/watch?v=jNQXAC9IVRw'],
        'maxItems': 5,
    }
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_youtube_transcript_scrape_empty_urls_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='video_urls'):
        client.youtube_transcript_scrape([])


# ---------------------------------------------------------------------------
# Client - media_transcribe
# ---------------------------------------------------------------------------


def test_media_transcribe_maps_input(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    _setup_run_and_items(mock_apify_client)

    run, items = client.media_transcribe(['https://example.com/episode.mp3'], max_results=3)

    mock_apify_client.actor.assert_called_once_with('steadyfetch/media-transcriber')
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'urls': ['https://example.com/episode.mp3']}
    assert mock_apify_client.dataset.return_value.list_items.call_args.kwargs['limit'] == 3
    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_media_transcribe_empty_urls_raises(client: ApifyToolsClient) -> None:
    with pytest.raises(ValueError, match='urls'):
        client.media_transcribe([])


# ---------------------------------------------------------------------------
# ApifyFacebookAdsTranscriptTool
# ---------------------------------------------------------------------------


def test_facebook_ads_transcript_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_ads_transcript_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyFacebookAdsTranscriptTool, mock_tools_client)

    result = tool._run(search_queries=['fitness app'], country='GB', max_results=12)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.facebook_ads_transcript_scrape.assert_called_once_with(
        search_queries=['fitness app'],
        country='GB',
        max_results=12,
        timeout_secs=600,
    )


def test_facebook_ads_transcript_tool_defaults_to_us(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_ads_transcript_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyFacebookAdsTranscriptTool, mock_tools_client)

    tool._run(search_queries=['fitness app'])

    assert mock_tools_client.facebook_ads_transcript_scrape.call_args.kwargs['country'] == 'US'


def test_facebook_ads_transcript_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.facebook_ads_transcript_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyFacebookAdsTranscriptTool, mock_tools_client, max_items=4)

    tool._run(search_queries=['fitness app'], max_results=500)

    assert mock_tools_client.facebook_ads_transcript_scrape.call_args.kwargs['max_results'] == 4


# ---------------------------------------------------------------------------
# ApifyYouTubeTranscriptTool
# ---------------------------------------------------------------------------


def test_youtube_transcript_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.youtube_transcript_scrape.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyYouTubeTranscriptTool, mock_tools_client)

    result = tool._run(video_urls=['https://www.youtube.com/watch?v=jNQXAC9IVRw'], max_results=5)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.youtube_transcript_scrape.assert_called_once_with(
        video_urls=['https://www.youtube.com/watch?v=jNQXAC9IVRw'],
        max_results=5,
        timeout_secs=600,
    )


def test_youtube_transcript_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.youtube_transcript_scrape.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyYouTubeTranscriptTool, mock_tools_client, max_items=2)

    tool._run(video_urls=['jNQXAC9IVRw'], max_results=900)

    assert mock_tools_client.youtube_transcript_scrape.call_args.kwargs['max_results'] == 2


# ---------------------------------------------------------------------------
# ApifyMediaTranscriberTool
# ---------------------------------------------------------------------------


def test_media_transcriber_tool_happy_path(mock_tools_client: MagicMock) -> None:
    mock_tools_client.media_transcribe.return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(ApifyMediaTranscriberTool, mock_tools_client)

    result = tool._run(urls=['https://example.com/episode.mp3'], max_results=3)
    parsed = json.loads(result)

    assert parsed['run'] == EXPECTED_RUN_META
    assert parsed['items'] == SAMPLE_ITEMS
    mock_tools_client.media_transcribe.assert_called_once_with(
        urls=['https://example.com/episode.mp3'],
        max_results=3,
        timeout_secs=600,
    )


def test_media_transcriber_tool_clamps_max_results(mock_tools_client: MagicMock) -> None:
    mock_tools_client.media_transcribe.return_value = (SUCCEEDED_RUN, [])
    tool = make_tool(ApifyMediaTranscriberTool, mock_tools_client, max_items=1)

    tool._run(urls=['https://example.com/episode.mp3'], max_results=50)

    assert mock_tools_client.media_transcribe.call_args.kwargs['max_results'] == 1


# ---------------------------------------------------------------------------
# Shared behaviour across the transcript family
# ---------------------------------------------------------------------------

# (tool_cls, client_method_name, _run kwargs)
_TRANSCRIPT_TOOL_INVOCATIONS: list[tuple[type, str, dict]] = [
    (ApifyFacebookAdsTranscriptTool, 'facebook_ads_transcript_scrape', {'search_queries': ['fitness app']}),
    (ApifyYouTubeTranscriptTool, 'youtube_transcript_scrape', {'video_urls': ['jNQXAC9IVRw']}),
    (ApifyMediaTranscriberTool, 'media_transcribe', {'urls': ['https://example.com/episode.mp3']}),
]


@pytest.mark.parametrize('tool_cls', APIFY_TRANSCRIPT_TOOLS)
def test_transcript_tool_handle_tool_error_enabled(tool_cls: type, mock_tools_client: MagicMock) -> None:
    tool = make_tool(tool_cls, mock_tools_client)
    assert tool.handle_tool_error is True


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TRANSCRIPT_TOOL_INVOCATIONS)
def test_transcript_tool_runtime_error_raises_tool_exception(
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


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TRANSCRIPT_TOOL_INVOCATIONS)
def test_transcript_tool_value_error_raises_tool_exception(
    tool_cls: type,
    method_name: str,
    run_kwargs: dict,
    mock_tools_client: MagicMock,
) -> None:
    getattr(mock_tools_client, method_name).side_effect = ValueError('At least one value is required.')
    tool = make_tool(tool_cls, mock_tools_client)

    with pytest.raises(ToolException, match='At least one value is required'):
        tool._run(**run_kwargs)


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TRANSCRIPT_TOOL_INVOCATIONS)
def test_transcript_tool_returns_valid_json_for_empty_items(
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


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TRANSCRIPT_TOOL_INVOCATIONS)
def test_transcript_tool_serialises_datetime_in_items(
    tool_cls: type,
    method_name: str,
    run_kwargs: dict,
    mock_tools_client: MagicMock,
) -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    items_with_datetime = [{'id': 'item-1', 'timestamp': timestamp, 'text': 'hello'}]
    getattr(mock_tools_client, method_name).return_value = (SUCCEEDED_RUN, items_with_datetime)
    tool = make_tool(tool_cls, mock_tools_client)

    result = tool._run(**run_kwargs)
    parsed = json.loads(result)

    assert isinstance(parsed['items'][0]['timestamp'], str)
    assert '2026-01-02' in parsed['items'][0]['timestamp']


@pytest.mark.parametrize(('tool_cls', 'method_name', 'run_kwargs'), _TRANSCRIPT_TOOL_INVOCATIONS)
def test_transcript_tool_envelope_shape(
    tool_cls: type,
    method_name: str,
    run_kwargs: dict,
    mock_tools_client: MagicMock,
) -> None:
    getattr(mock_tools_client, method_name).return_value = (SUCCEEDED_RUN, SAMPLE_ITEMS)
    tool = make_tool(tool_cls, mock_tools_client)

    parsed = json.loads(tool._run(**run_kwargs))

    assert set(parsed) == {'run', 'items'}
    assert isinstance(parsed['items'], list)
    assert isinstance(parsed['run'], dict)


def test_transcript_tool_group_membership() -> None:
    expected = [ApifyFacebookAdsTranscriptTool, ApifyYouTubeTranscriptTool, ApifyMediaTranscriberTool]
    assert expected == APIFY_TRANSCRIPT_TOOLS


@pytest.mark.parametrize('tool_cls', APIFY_TRANSCRIPT_TOOLS)
def test_transcript_tool_description_carries_no_price(tool_cls: type) -> None:
    """Prices go stale and cannot be edited after a release, so they stay out of the package."""
    description = tool_cls.model_fields['description'].default
    assert '$' not in description
    assert 'price' not in description.lower()
