"""Transcription Actor tools and their input schemas.

Each tool wraps a single Apify transcription Actor behind a simplified,
LLM-friendly interface returning the standard JSON envelope. The other tool
families return what a page or profile *shows*; these return the spoken words
inside a video or audio file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import ArgsSchema, ToolException
from pydantic import BaseModel, Field

from langchain_apify._constants import (
    _DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
    _MAX_ITEMS_CAP,
)
from langchain_apify.tools.base import _TOOL_RUN_ERRORS, _ApifyGenericTool

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class ApifyFacebookAdsTranscriptInput(BaseModel):
    """Input schema for :class:`ApifyFacebookAdsTranscriptTool`."""

    search_queries: list[str] = Field(
        description=(
            'Keywords or advertiser page names to search the Facebook Ad Library '
            'for (e.g. ["fitness app"]). At least one value is required.'
        ),
    )
    country: str = Field(
        default='US',
        description='Two-letter country code the ads are served in (e.g. "US", "GB", "DE").',
    )
    max_results: int = Field(
        default=_DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        description=f'Maximum number of ad creatives to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


class ApifyYouTubeTranscriptInput(BaseModel):
    """Input schema for :class:`ApifyYouTubeTranscriptTool`."""

    video_urls: list[str] = Field(
        description=(
            'YouTube watch, youtu.be, /shorts/ or /live/ links, or bare 11-character '
            'video IDs (e.g. ["https://www.youtube.com/watch?v=jNQXAC9IVRw"]). '
            'At least one value is required.'
        ),
    )
    max_results: int = Field(
        default=_DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        description=f'Maximum number of transcripts to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


class ApifyMediaTranscriberInput(BaseModel):
    """Input schema for :class:`ApifyMediaTranscriberTool`."""

    urls: list[str] = Field(
        description=(
            'Direct audio or video file links on any host, or page links on a supported '
            'podcast or video host (e.g. ["https://example.com/episode.mp3"]). '
            'At least one value is required.'
        ),
    )
    max_results: int = Field(
        default=_DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        description=f'Maximum number of transcripts to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ApifyFacebookAdsTranscriptTool(_ApifyGenericTool):  # type: ignore[override]
    """Transcribe the Facebook Ad Library ads running now for a keyword or advertiser.

    Uses the ``steadyfetch/facebook-ads-transcript-scraper`` Actor under the
    hood: each matching ad creative comes back with its video transcript, the
    opening hook, the call to action and the ad's metadata (on-image text for
    image ads).

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of ad dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyFacebookAdsTranscriptTool

            tool = ApifyFacebookAdsTranscriptTool()
            result = tool.invoke({
                "search_queries": ["fitness app"],
                "country": "US",
                "max_results": 10,
            })
    """

    name: str = 'apify_facebook_ads_transcript'
    description: str = (
        'Transcribe the Facebook Ad Library ads running now for a keyword or advertiser page name'
        ' and return the transcript, opening hook, call to action and ad metadata as JSON.'
        ' Required: search_queries (list of str - keywords or advertiser page names, e.g. ["fitness app"]).'
        ' Optional: country (str - two-letter code, default "US"),'
        f' max_results (int, default {_DEFAULT_TRANSCRIPT_RESULTS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyFacebookAdsTranscriptInput

    def _run(
        self,
        search_queries: list[str],
        country: str = 'US',
        max_results: int = _DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.facebook_ads_transcript_scrape(
                search_queries=search_queries,
                country=country,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except _TOOL_RUN_ERRORS as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyYouTubeTranscriptTool(_ApifyGenericTool):  # type: ignore[override]
    """Get the transcript of specific YouTube videos.

    Uses the ``steadyfetch/youtube-transcript-scraper`` Actor under the hood:
    published captions are used where they exist, and speech-to-text fills in
    the videos that have none.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of transcript dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyYouTubeTranscriptTool

            tool = ApifyYouTubeTranscriptTool()
            result = tool.invoke({
                "video_urls": ["https://www.youtube.com/watch?v=jNQXAC9IVRw"],
            })
    """

    name: str = 'apify_youtube_transcript'
    description: str = (
        'Get the transcript of specific YouTube videos and return them as JSON.'
        ' Published captions are used where they exist; videos without captions are transcribed by speech-to-text.'
        ' Required: video_urls (list of str - watch, youtu.be, /shorts/ or /live/ links,'
        ' or bare 11-character video IDs).'
        f' Optional: max_results (int, default {_DEFAULT_TRANSCRIPT_RESULTS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyYouTubeTranscriptInput

    def _run(
        self,
        video_urls: list[str],
        max_results: int = _DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.youtube_transcript_scrape(
                video_urls=video_urls,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except _TOOL_RUN_ERRORS as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyMediaTranscriberTool(_ApifyGenericTool):  # type: ignore[override]
    """Transcribe audio or video from a file link or a supported media page.

    Uses the ``steadyfetch/media-transcriber`` Actor under the hood: a direct
    audio or video file link on any host, or a page link on a supported
    podcast or video host (Libsyn, Megaphone, Buzzsprout, Acast, Apple
    Podcasts, Spotify for Creators, Archive.org, SoundCloud, Loom, Twitch
    VODs, Wistia, Facebook and TikTok video links).

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of transcript dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyMediaTranscriberTool

            tool = ApifyMediaTranscriberTool()
            result = tool.invoke({
                "urls": ["https://example.com/episode.mp3"],
            })
    """

    name: str = 'apify_media_transcriber'
    description: str = (
        'Transcribe audio or video and return the text as JSON.'
        ' Accepts a direct audio or video file link on any host, or a page link on a supported podcast'
        ' or video host (Libsyn, Megaphone, Buzzsprout, Acast, Apple Podcasts, Spotify for Creators,'
        ' Archive.org, SoundCloud, Loom, Twitch VODs, Wistia, Facebook and TikTok video links).'
        ' Required: urls (list of str).'
        f' Optional: max_results (int, default {_DEFAULT_TRANSCRIPT_RESULTS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' For YouTube links use apify_youtube_transcript instead.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyMediaTranscriberInput

    def _run(
        self,
        urls: list[str],
        max_results: int = _DEFAULT_TRANSCRIPT_RESULTS_LIMIT,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.media_transcribe(
                urls=urls,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except _TOOL_RUN_ERRORS as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


# Convenience tool-class list for selective agent binding.
APIFY_TRANSCRIPT_TOOLS: list[type[BaseTool]] = [
    ApifyFacebookAdsTranscriptTool,
    ApifyYouTubeTranscriptTool,
    ApifyMediaTranscriberTool,
]
