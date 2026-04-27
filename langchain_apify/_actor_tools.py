"""Apify Actor-specific LangChain tools for social media platforms.

Each tool wraps a single Apify Actor behind a simplified, LLM-friendly
interface so that LangChain agents can scrape social media data without
needing to know Actor IDs or raw input schemas.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_apify.tools import _ApifyGenericTool, _run_meta

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class ApifyInstagramScraperInput(BaseModel):
    """Input schema for :class:`ApifyInstagramScraperTool`."""

    search_type: Literal['user', 'hashtag', 'post', 'comments'] = Field(
        description=(
            'Type of data to scrape: "user" for a profile\'s posts, "hashtag" '
            'for posts under a tag, "post" for a single post, "comments" for '
            'comments on a post.'
        ),
    )
    search_query: str = Field(
        description=(
            'Username, hashtag, or full Instagram URL depending on search_type. '
            'For "comments" you must pass a post URL (e.g. instagram.com/p/...).'
        ),
    )
    max_results: int = Field(default=20, description='Maximum number of items to return.')
    only_posts_newer_than: str | None = Field(
        default=None,
        description=(
            'Optional date filter. Accepts YYYY-MM-DD, ISO-8601, or relative '
            'values like "1 day", "2 months", "3 years".'
        ),
    )


class ApifyLinkedInProfilePostsInput(BaseModel):
    """Input schema for :class:`ApifyLinkedInProfilePostsTool`."""

    profile_url: str = Field(
        description='LinkedIn profile URL or username (e.g. "satyanadella" or "linkedin.com/in/satyanadella").',
    )
    max_results: int = Field(default=20, description='Maximum number of posts to return.')


class ApifyLinkedInProfileSearchInput(BaseModel):
    """Input schema for :class:`ApifyLinkedInProfileSearchTool`."""

    query: str = Field(description='Search keywords (e.g. name, title, company).')
    max_results: int = Field(default=10, description='Maximum number of profiles to return.')


class ApifyLinkedInProfileDetailInput(BaseModel):
    """Input schema for :class:`ApifyLinkedInProfileDetailTool`."""

    profile_url: str = Field(
        description='LinkedIn profile URL, username, or URN (e.g. "neal-mohan").',
    )
    include_email: bool = Field(
        default=False,
        description='If True, attempt to include the profile email when available.',
    )


class ApifyTwitterScraperInput(BaseModel):
    """Input schema for :class:`ApifyTwitterScraperTool`."""

    search_query: str = Field(description='Search term, Twitter handle, or tweet URL.')
    search_mode: Literal['search', 'user', 'replies'] = Field(
        default='search',
        description=(
            'Scraping mode: "search" for keyword search, "user" for a handle\'s '
            'tweets, "replies" for a tweet URL\'s replies.'
        ),
    )
    max_results: int = Field(default=20, description='Maximum number of tweets to return.')
    start: str | None = Field(
        default=None,
        description='Optional start date — only return tweets newer than this date.',
    )
    end: str | None = Field(
        default=None,
        description='Optional end date — only return tweets older than this date.',
    )


class ApifyTikTokScraperInput(BaseModel):
    """Input schema for :class:`ApifyTikTokScraperTool`."""

    search_query: str = Field(description='Username, hashtag, or search keyword.')
    search_type: Literal['search', 'user', 'hashtag'] = Field(
        default='search',
        description=(
            'Type of content to scrape: "search" for keyword search, "user" for '
            "a profile's videos, \"hashtag\" for videos under a tag."
        ),
    )
    max_results: int = Field(default=20, description='Maximum number of items to return.')


class ApifyFacebookPostsScraperInput(BaseModel):
    """Input schema for :class:`ApifyFacebookPostsScraperTool`."""

    page_url: str = Field(description='Facebook page URL to scrape (public pages only).')
    max_results: int = Field(default=20, description='Maximum number of posts to return.')
    only_posts_newer_than: str | None = Field(
        default=None,
        description=(
            'Optional date filter. Accepts YYYY-MM-DD, ISO-8601, or relative '
            'values like "1 day", "2 months", "3 years".'
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ApifyInstagramScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape Instagram profiles, hashtags, posts, or comments.

    Uses the ``apify/instagram-scraper`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of scraped item dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyInstagramScraperTool

            tool = ApifyInstagramScraperTool()
            result = tool.invoke({
                "search_type": "user",
                "search_query": "apify",
                "max_results": 10,
            })
    """

    name: str = 'apify_instagram_scraper'
    description: str = (
        'Scrape Instagram profiles, hashtags, posts, or comments and return the results as JSON.'
        ' Required: search_type (one of "user", "hashtag", "post", "comments"),'
        ' search_query (str — username, hashtag, or post URL).'
        ' Optional: max_results (int, default 20),'
        ' only_posts_newer_than (str — date filter, e.g. "2025-01-01" or "1 week").'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyInstagramScraperInput

    def _run(
        self,
        search_type: Literal['user', 'hashtag', 'post', 'comments'],
        search_query: str,
        max_results: int = 20,
        only_posts_newer_than: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.instagram_scrape(
                search_type=search_type,
                search_query=search_query,
                max_results=self._clamp_items(max_results),
                only_posts_newer_than=only_posts_newer_than,
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})
