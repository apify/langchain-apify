"""Apify Actor-specific LangChain tools for social media platforms.

Each tool wraps a single Apify Actor behind a simplified, LLM-friendly
interface so that LangChain agents can scrape social media data without
needing to know Actor IDs or raw input schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
