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
        description='Optional start date - only return tweets newer than this date.',
    )
    end: str | None = Field(
        default=None,
        description='Optional end date - only return tweets older than this date.',
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
        ' search_query (str - username, hashtag, or post URL).'
        ' Optional: max_results (int, default 20),'
        ' only_posts_newer_than (str - date filter, e.g. "2025-01-01" or "1 week").'
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


class ApifyLinkedInProfilePostsTool(_ApifyGenericTool):  # type: ignore[override]
    """Extract posts from a LinkedIn profile.

    Uses the ``apimaestro/linkedin-profile-posts`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of post dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyLinkedInProfilePostsTool

            tool = ApifyLinkedInProfilePostsTool()
            result = tool.invoke({
                "profile_url": "https://www.linkedin.com/in/satyanadella",
                "max_results": 10,
            })
    """

    name: str = 'apify_linkedin_profile_posts'
    description: str = (
        'Extract posts from a LinkedIn profile and return them as JSON.'
        ' Required: profile_url (str - LinkedIn profile URL or username, e.g. "satyanadella").'
        ' Optional: max_results (int, default 20).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfilePostsInput

    def _run(
        self,
        profile_url: str,
        max_results: int = 20,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_posts(
                profile_url=profile_url,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyLinkedInProfileSearchTool(_ApifyGenericTool):  # type: ignore[override]
    """Search for LinkedIn profiles by keyword or criteria.

    Uses the ``harvestapi/linkedin-profile-search`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of profile dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyLinkedInProfileSearchTool

            tool = ApifyLinkedInProfileSearchTool()
            result = tool.invoke({
                "query": "Founder",
                "max_results": 10,
            })
    """

    name: str = 'apify_linkedin_profile_search'
    description: str = (
        'Search for LinkedIn profiles by keyword (name, title, company) and return matching profiles as JSON.'
        ' Required: query (str - search keywords).'
        ' Optional: max_results (int, default 10).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfileSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_search(
                query=query,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyLinkedInProfileDetailTool(_ApifyGenericTool):  # type: ignore[override]
    """Retrieve detailed information from a specific LinkedIn profile.

    Uses the ``apimaestro/linkedin-profile-detail`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (typically
        a single-element list with the profile dict).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyLinkedInProfileDetailTool

            tool = ApifyLinkedInProfileDetailTool()
            result = tool.invoke({
                "profile_url": "https://www.linkedin.com/in/neal-mohan",
            })
    """

    name: str = 'apify_linkedin_profile_detail'
    description: str = (
        'Retrieve detailed information from a specific LinkedIn profile and return it as JSON.'
        ' Required: profile_url (str - LinkedIn profile URL, username, or URN, e.g. "neal-mohan").'
        ' Optional: include_email (bool, default False - include profile email if available).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfileDetailInput

    def _run(
        self,
        profile_url: str,
        include_email: bool = False,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_detail(
                profile_url=profile_url,
                include_email=include_email,
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyTwitterScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape tweets, profiles, or replies from Twitter/X.

    Uses the ``apidojo/twitter-scraper-lite`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of tweet dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyTwitterScraperTool

            tool = ApifyTwitterScraperTool()
            result = tool.invoke({
                "search_query": "apify",
                "search_mode": "search",
                "max_results": 20,
            })
    """

    name: str = 'apify_twitter_scraper'
    description: str = (
        'Scrape tweets from Twitter/X by search term, user handle, or tweet URL and return them as JSON.'
        ' Required: search_query (str - search term, handle, or tweet URL).'
        ' Optional: search_mode (one of "search", "user", "replies"; default "search"),'
        ' max_results (int, default 20),'
        ' start (str - ISO date, only return tweets newer than this date),'
        ' end (str - ISO date, only return tweets older than this date).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyTwitterScraperInput

    def _run(
        self,
        search_query: str,
        search_mode: Literal['search', 'user', 'replies'] = 'search',
        max_results: int = 20,
        start: str | None = None,
        end: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.twitter_scrape(
                search_query=search_query,
                search_mode=search_mode,
                max_results=self._clamp_items(max_results),
                start=start,
                end=end,
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyTikTokScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape TikTok videos, profiles, or hashtag content.

    Uses the ``clockworks/tiktok-scraper`` Actor under the hood.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of TikTok item dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyTikTokScraperTool

            tool = ApifyTikTokScraperTool()
            result = tool.invoke({
                "search_query": "cooking",
                "search_type": "search",
                "max_results": 20,
            })
    """

    name: str = 'apify_tiktok_scraper'
    description: str = (
        'Scrape TikTok by search keyword, profile, or hashtag and return the results as JSON.'
        ' Required: search_query (str - keyword, username, or hashtag).'
        ' Optional: search_type (one of "search", "user", "hashtag"; default "search"),'
        ' max_results (int, default 20).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyTikTokScraperInput

    def _run(
        self,
        search_query: str,
        search_type: Literal['search', 'user', 'hashtag'] = 'search',
        max_results: int = 20,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.tiktok_scrape(
                search_query=search_query,
                search_type=search_type,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyFacebookPostsScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape public Facebook page posts.

    Uses the ``apify/facebook-posts-scraper`` Actor under the hood.
    Only public Facebook pages are supported - personal profiles cannot
    be scraped.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of post dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyFacebookPostsScraperTool

            tool = ApifyFacebookPostsScraperTool()
            result = tool.invoke({
                "page_url": "https://www.facebook.com/humansofnewyork/",
                "max_results": 20,
            })
    """

    name: str = 'apify_facebook_posts_scraper'
    description: str = (
        'Scrape posts from a public Facebook page and return them as JSON.'
        ' Required: page_url (str - Facebook page URL; personal profiles are not supported).'
        ' Optional: max_results (int, default 20),'
        ' only_posts_newer_than (str - date filter, e.g. "2025-01-01" or "1 week").'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
    )
    args_schema: type[BaseModel] = ApifyFacebookPostsScraperInput

    def _run(
        self,
        page_url: str,
        max_results: int = 20,
        only_posts_newer_than: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.facebook_posts_scrape(
                page_url=page_url,
                max_results=self._clamp_items(max_results),
                only_posts_newer_than=only_posts_newer_than,
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})
