"""Actor-specific tool subclasses.

Tools in this module wrap a single Apify Actor behind a simplified,
LLM-friendly interface. They inherit from
:class:`~langchain_apify.tools._ApifyGenericTool`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_apify._client import (
    _DEFAULT_CRAWLER_TYPE,
    _DEFAULT_GOOGLE_MAX_RESULTS,
    _DEFAULT_LINKEDIN_SEARCH_MAX_RESULTS,
    _DEFAULT_MAX_CRAWL_DEPTH,
    _DEFAULT_MAX_CRAWL_PAGES,
    _DEFAULT_RAG_MAX_RESULTS,
    _DEFAULT_RUN_TIMEOUT_SECS,
    _DEFAULT_SOCIAL_RESULTS_LIMIT,
)
from langchain_apify._types import CrawlerType  # noqa: TCH001  # runtime-needed: shared Literal alias
from langchain_apify._utils import _extract_content, _extract_source, _safe_title
from langchain_apify.tools import (
    ApifyGoogleSearchInput,
    ApifyWebCrawlerInput,
    _ApifyGenericTool,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun

# Per-tool result limits not shared via ``_client`` (Maps/YouTube/Ecommerce).
_DEFAULT_GOOGLE_MAPS_MAX_RESULTS = 10
_DEFAULT_YOUTUBE_MAX_RESULTS = 10
_DEFAULT_ECOMMERCE_MAX_RESULTS = 20

# Shared Literal aliases so each social tool declares its accepted values once
# (used in both the input schema and the tool ``_run`` signature).
InstagramSearchType = Literal['user', 'hashtag', 'post', 'comments']
TwitterSearchMode = Literal['search', 'user', 'replies']
TwitterSort = Literal['Latest', 'Top']
TikTokSearchType = Literal['search', 'user', 'hashtag', 'post']


# ---------------------------------------------------------------------------
# Search & Crawling tools
# ---------------------------------------------------------------------------


class ApifyGoogleSearchTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google and return structured results via Apify.

    Wraps the ``apify/google-search-scraper`` Actor behind a simplified,
    LLM-friendly interface.  Returns a JSON envelope whose ``items`` are
    result objects, each with ``title``, ``url``, and ``description`` keys.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": null, "items": [{"title", "url", "description"}]}``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyGoogleSearchTool

            tool = ApifyGoogleSearchTool()
            results = tool.invoke({"query": "LangChain framework"})
    """

    name: str = 'apify_google_search'
    description: str = (
        'Search Google using Apify and return a JSON envelope.'
        ' Each item has keys: title, url, description.'
        ' Required: query (str) — the search query.'
        f' Optional: max_results (int, default {_DEFAULT_GOOGLE_MAX_RESULTS}),'
        ' country_code (str|null), language_code (str|null),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns JSON with keys: run (null), items.'
    )
    args_schema: type[BaseModel] = ApifyGoogleSearchInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_GOOGLE_MAX_RESULTS,
        country_code: str | None = None,
        language_code: str | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            results = self._client.google_search(
                query,
                max_results=self._clamp_items(max_results),
                country_code=country_code,
                language_code=language_code,
                timeout_secs=self._clamp_timeout(timeout_secs),
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        # default=str coerces any non-JSON-native types (e.g. datetime from
        # the Apify client's clean=True deserialiser) to their string repr
        # so the LLM never sees a serialisation failure.
        return self._envelope(None, results)


class ApifyWebCrawlerTool(_ApifyGenericTool):  # type: ignore[override]
    """Crawl a website and return page content as JSON via Apify.

    Wraps the ``apify/website-content-crawler`` Actor.  Returns a JSON envelope
    whose ``items`` are page objects, each with ``url``, ``title``, and
    ``content`` (markdown) keys.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": null, "items": [{"url", "title", "content"}]}``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyWebCrawlerTool

            tool = ApifyWebCrawlerTool()
            pages = tool.invoke({
                "url": "https://docs.apify.com",
                "max_crawl_pages": 5,
            })
    """

    name: str = 'apify_web_crawler'
    description: str = (
        'Crawl a website using Apify and return a JSON envelope.'
        ' Each item has keys: url, title, content (markdown).'
        ' Required: url (str) — seed URL to crawl.'
        f' Optional: max_crawl_pages (int, default {_DEFAULT_MAX_CRAWL_PAGES}),'
        f' max_crawl_depth (int, default {_DEFAULT_MAX_CRAWL_DEPTH}),'
        f' crawler_type (str, default "{_DEFAULT_CRAWLER_TYPE}"),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns JSON with keys: run (null), items.'
    )
    args_schema: type[BaseModel] = ApifyWebCrawlerInput

    def _run(
        self,
        url: str,
        max_crawl_pages: int = _DEFAULT_MAX_CRAWL_PAGES,
        max_crawl_depth: int = _DEFAULT_MAX_CRAWL_DEPTH,
        crawler_type: CrawlerType = _DEFAULT_CRAWLER_TYPE,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            items = self._client.crawl_website(
                url,
                max_crawl_pages=self._clamp_items(max_crawl_pages),
                max_crawl_depth=self._clamp_depth(max_crawl_depth),
                crawler_type=crawler_type,
                timeout_secs=self._clamp_timeout(timeout_secs),
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        # Defensive filter: some Actor responses occasionally surface list-typed
        # entries (e.g. nested arrays for sitemap-style outputs). Skip anything
        # that isn't a dict so .get() never blows up.
        pages = [
            {
                'url': item.get('url', ''),
                'title': _safe_title(item),
                'content': _extract_content(item),
            }
            for item in items
            if isinstance(item, dict)
        ]
        return self._envelope(None, pages)


# ---------------------------------------------------------------------------
# Input schemas (US-4 Search & Crawling Actor tools)
# ---------------------------------------------------------------------------


class ApifyRAGWebBrowserInput(BaseModel):
    """Input schema for :class:`ApifyRAGWebBrowserTool`."""

    query: str = Field(description='Search query string.')
    max_results: int = Field(default=_DEFAULT_RAG_MAX_RESULTS, description='Maximum number of results to return.')


class ApifyGoogleMapsInput(BaseModel):
    """Input schema for :class:`ApifyGoogleMapsTool`."""

    query: str = Field(description='Search query (e.g. "coffee shops in Berlin").')
    max_results: int = Field(
        default=_DEFAULT_GOOGLE_MAPS_MAX_RESULTS, description='Maximum number of places to return.'
    )
    language: str | None = Field(
        default=None,
        description='Optional ISO language code for results (e.g. "en", "de").',
    )


class ApifyYouTubeScraperInput(BaseModel):
    """Input schema for :class:`ApifyYouTubeScraperTool`."""

    search_query: str = Field(
        description=('Keyword for "search" mode, or a video/channel URL for "video"/"channel" modes.'),
    )
    search_type: Literal['search', 'video', 'channel'] = Field(
        default='search',
        description='Scrape mode: search keyword, single video URL, or channel URL.',
    )
    max_results: int = Field(default=_DEFAULT_YOUTUBE_MAX_RESULTS, description='Maximum number of items to return.')


class ApifyEcommerceScraperInput(BaseModel):
    """Input schema for :class:`ApifyEcommerceScraperTool`."""

    url: str = Field(description='Product-detail URL or category / listing page URL to scrape.')
    url_type: Literal['product', 'category'] = Field(
        default='product',
        description=(
            'Type of page the URL points to: "product" for a product-detail page, '
            '"category" for a category / listing page.'
        ),
    )
    max_results: int = Field(
        default=_DEFAULT_ECOMMERCE_MAX_RESULTS, description='Maximum number of products to return.'
    )


# ---------------------------------------------------------------------------
# Tools (US-4 Search & Crawling Actor tools)
# ---------------------------------------------------------------------------


class ApifyRAGWebBrowserTool(_ApifyGenericTool):  # type: ignore[override]
    """Search the web and return content from top results.

    Wraps the ``apify/rag-web-browser`` Actor.  Unlike
    :class:`ApifySearchRetriever` (which returns LangChain ``Document``
    objects for RAG pipelines), this tool returns a JSON envelope
    suitable for agent tool-calling.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [{"url", "title", "content"}]}``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyRAGWebBrowserTool

            tool = ApifyRAGWebBrowserTool()
            result = tool.invoke({"query": "what is LangChain?", "max_results": 3})
    """

    name: str = 'apify_rag_web_browser'
    description: str = (
        'Search the web and return a JSON envelope with crawled results.'
        ' Each item has keys: url, title, content.'
        ' Required: query (str) - the search query.'
        f' Optional: max_results (int, default {_DEFAULT_RAG_MAX_RESULTS}).'
        ' Returns keys: run, items.'
    )
    args_schema: type[BaseModel] = ApifyRAGWebBrowserInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_RAG_MAX_RESULTS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.rag_web_search(
                query,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        results = [
            {
                'url': _extract_source(item),
                'title': _safe_title(item),
                'content': _extract_content(item),
            }
            for item in items
            if isinstance(item, dict)
        ]
        return self._envelope(run, results)


class ApifyGoogleMapsTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google Maps for places, reviews, and business details.

    Wraps the ``compass/crawler-google-places`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [...]}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``
        and ``items`` are place dicts.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyGoogleMapsTool

            tool = ApifyGoogleMapsTool()
            result = tool.invoke({"query": "coffee shops in Berlin", "max_results": 5})
    """

    name: str = 'apify_google_maps'
    description: str = (
        'Search Google Maps places, reviews, and business details and return a JSON envelope.'
        ' Required: query (str) - the search query.'
        f' Optional: max_results (int, default {_DEFAULT_GOOGLE_MAPS_MAX_RESULTS}),'
        ' language (str|null - ISO code, e.g. "en").'
        ' Returns keys: run, items.'
    )
    args_schema: type[BaseModel] = ApifyGoogleMapsInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_GOOGLE_MAPS_MAX_RESULTS,
        language: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.google_maps_search(
                query,
                max_results=self._clamp_items(max_results),
                language=language,
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyYouTubeScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape YouTube videos, channels, or search results.

    Wraps the ``streamers/youtube-scraper`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [...]}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``
        and ``items`` are video / channel dicts.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyYouTubeScraperTool

            tool = ApifyYouTubeScraperTool()
            result = tool.invoke({
                "search_query": "langchain tutorial",
                "search_type": "search",
                "max_results": 5,
            })
    """

    name: str = 'apify_youtube_scraper'
    description: str = (
        'Scrape YouTube by keyword, video URL, or channel URL and return a JSON envelope.'
        ' Required: search_query (str - keyword for "search" mode, or a video/channel URL).'
        ' Optional: search_type (one of "search", "video", "channel"; default "search"),'
        f' max_results (int, default {_DEFAULT_YOUTUBE_MAX_RESULTS}).'
        ' Returns keys: run, items.'
    )
    args_schema: type[BaseModel] = ApifyYouTubeScraperInput

    def _run(
        self,
        search_query: str,
        search_type: Literal['search', 'video', 'channel'] = 'search',
        max_results: int = _DEFAULT_YOUTUBE_MAX_RESULTS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.youtube_scrape(
                search_query=search_query,
                search_type=search_type,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyEcommerceScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Extract product or listing data from an e-commerce URL.

    Wraps the ``apify/e-commerce-scraping-tool`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [...]}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``
        and ``items`` are product / listing dicts.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyEcommerceScraperTool

            tool = ApifyEcommerceScraperTool()
            result = tool.invoke({
                "url": "https://shop.example.com/category/123",
                "url_type": "category",
                "max_results": 20,
            })
    """

    name: str = 'apify_ecommerce_scraper'
    description: str = (
        'Extract product data from an e-commerce URL and return a JSON envelope.'
        ' Required: url (str) - product-detail or category / listing URL.'
        ' Optional: url_type (one of "product", "category"; default "product"),'
        f' max_results (int, default {_DEFAULT_ECOMMERCE_MAX_RESULTS}).'
        ' Returns keys: run, items.'
    )
    args_schema: type[BaseModel] = ApifyEcommerceScraperInput

    def _run(
        self,
        url: str,
        url_type: Literal['product', 'category'] = 'product',
        max_results: int = _DEFAULT_ECOMMERCE_MAX_RESULTS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.ecommerce_scrape(
                url,
                url_type=url_type,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class ApifyInstagramScraperInput(BaseModel):
    """Input schema for :class:`ApifyInstagramScraperTool`."""

    search_type: InstagramSearchType = Field(
        description=(
            'Type of data to scrape: "user" for a profile\'s posts, "hashtag" '
            'for posts under a tag, "post" for a single post, "comments" for '
            'comments on a post.'
        ),
    )
    search_query: str = Field(
        description=(
            'Username, hashtag, or a full Instagram URL including the scheme '
            '(e.g. https://www.instagram.com/p/ABC123/). For "comments" you must '
            'pass a full post URL.'
        ),
    )
    max_results: int = Field(default=_DEFAULT_SOCIAL_RESULTS_LIMIT, description='Maximum number of items to return.')
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
    max_results: int = Field(default=_DEFAULT_SOCIAL_RESULTS_LIMIT, description='Maximum number of posts to return.')


class ApifyLinkedInProfileSearchInput(BaseModel):
    """Input schema for :class:`ApifyLinkedInProfileSearchTool`."""

    query: str = Field(description='Search keywords (e.g. name, title, company).')
    max_results: int = Field(
        default=_DEFAULT_LINKEDIN_SEARCH_MAX_RESULTS, description='Maximum number of profiles to return.'
    )


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
    search_mode: TwitterSearchMode = Field(
        default='search',
        description=(
            'Scraping mode: "search" for keyword search, "user" for a handle\'s '
            'tweets, "replies" for a tweet URL\'s replies.'
        ),
    )
    max_results: int = Field(default=_DEFAULT_SOCIAL_RESULTS_LIMIT, description='Maximum number of tweets to return.')
    start: str | None = Field(
        default=None,
        description='Optional start date - only return tweets newer than this date.',
    )
    end: str | None = Field(
        default=None,
        description='Optional end date - only return tweets older than this date.',
    )
    sort: TwitterSort | None = Field(
        default=None,
        description='Optional sort order: "Latest" for most recent first, "Top" for most popular.',
    )


class ApifyTikTokScraperInput(BaseModel):
    """Input schema for :class:`ApifyTikTokScraperTool`."""

    search_query: str = Field(description='Username, hashtag, search keyword, or TikTok post URL.')
    search_type: TikTokSearchType = Field(
        default='search',
        description=(
            'Type of content to scrape: "search" for keyword search, "user" for '
            'a profile\'s videos, "hashtag" for videos under a tag, "post" for a '
            'specific TikTok post URL.'
        ),
    )
    max_results: int = Field(default=_DEFAULT_SOCIAL_RESULTS_LIMIT, description='Maximum number of items to return.')


class ApifyFacebookPostsScraperInput(BaseModel):
    """Input schema for :class:`ApifyFacebookPostsScraperTool`."""

    page_url: str = Field(description='Facebook page URL to scrape (public pages only).')
    max_results: int = Field(default=_DEFAULT_SOCIAL_RESULTS_LIMIT, description='Maximum number of posts to return.')
    only_posts_newer_than: str | None = Field(
        default=None,
        description=(
            'Optional date filter. Accepts YYYY-MM-DD, ISO-8601, or relative '
            'values like "1 day", "2 months", "3 years".'
        ),
    )
    only_posts_older_than: str | None = Field(
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
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of scraped item dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        ' search_query (str - username, hashtag, or a full Instagram URL including the scheme,'
        ' e.g. https://www.instagram.com/p/ABC123/; "comments" requires a full post URL).'
        f' Optional: max_results (int, default {_DEFAULT_SOCIAL_RESULTS_LIMIT}),'
        ' only_posts_newer_than (str - date filter, e.g. "2025-01-01" or "1 week").'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyInstagramScraperInput

    def _run(
        self,
        search_type: InstagramSearchType,
        search_query: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
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
        return self._envelope(run, items)


class ApifyLinkedInProfilePostsTool(_ApifyGenericTool):  # type: ignore[override]
    """Extract posts from a LinkedIn profile.

    Uses the ``apimaestro/linkedin-profile-posts`` Actor under the hood.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of post dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        f' Optional: max_results (int, default {_DEFAULT_SOCIAL_RESULTS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfilePostsInput

    def _run(
        self,
        profile_url: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_posts(
                profile_url=profile_url,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyLinkedInProfileSearchTool(_ApifyGenericTool):  # type: ignore[override]
    """Search for LinkedIn profiles by keyword or criteria.

    Uses the ``harvestapi/linkedin-profile-search`` Actor under the hood.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of profile dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        f' Optional: max_results (int, default {_DEFAULT_LINKEDIN_SEARCH_MAX_RESULTS}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfileSearchInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_LINKEDIN_SEARCH_MAX_RESULTS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_search(
                query=query,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyLinkedInProfileDetailTool(_ApifyGenericTool):  # type: ignore[override]
    """Retrieve detailed information from a specific LinkedIn profile.

    Uses the ``apimaestro/linkedin-profile-detail`` Actor under the hood.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (typically
        a single-element list with the profile dict).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyLinkedInProfileDetailInput

    def _run(
        self,
        profile_url: str,
        *,
        include_email: bool = False,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.linkedin_profile_detail(
                profile_url=profile_url,
                include_email=include_email,
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyTwitterScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape tweets, profiles, or replies from Twitter/X.

    Uses the ``apidojo/twitter-scraper-lite`` Actor under the hood.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of tweet dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        f' max_results (int, default {_DEFAULT_SOCIAL_RESULTS_LIMIT}),'
        ' start (str - ISO date, only return tweets newer than this date),'
        ' end (str - ISO date, only return tweets older than this date),'
        ' sort (one of "Latest", "Top" - sort order for results).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyTwitterScraperInput

    def _run(  # noqa: PLR0913
        self,
        search_query: str,
        search_mode: TwitterSearchMode = 'search',
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        start: str | None = None,
        end: str | None = None,
        sort: TwitterSort | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.twitter_scrape(
                search_query=search_query,
                search_mode=search_mode,
                max_results=self._clamp_items(max_results),
                start=start,
                end=end,
                sort=sort,
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


class ApifyTikTokScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape TikTok videos, profiles, or hashtag content.

    Uses the ``clockworks/tiktok-scraper`` Actor under the hood.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of TikTok item dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        'Scrape TikTok by search keyword, profile, hashtag, or post URL and return the results as JSON.'
        ' Required: search_query (str - keyword, username, hashtag, or TikTok post URL).'
        ' Optional: search_type (one of "search", "user", "hashtag", "post"; default "search"),'
        f' max_results (int, default {_DEFAULT_SOCIAL_RESULTS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyTikTokScraperInput

    def _run(
        self,
        search_query: str,
        search_type: TikTokSearchType = 'search',
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
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
        return self._envelope(run, items)


class ApifyFacebookPostsScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape public Facebook page posts.

    Uses the ``apify/facebook-posts-scraper`` Actor under the hood.
    Only public Facebook pages are supported - personal profiles cannot
    be scraped.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of post dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        f' Optional: max_results (int, default {_DEFAULT_SOCIAL_RESULTS_LIMIT}),'
        ' only_posts_newer_than (str - date filter, e.g. "2025-01-01" or "1 week"),'
        ' only_posts_older_than (str - date filter, e.g. "2025-01-01" or "1 week").'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyFacebookPostsScraperInput

    def _run(
        self,
        page_url: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        only_posts_newer_than: str | None = None,
        only_posts_older_than: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.facebook_posts_scrape(
                page_url=page_url,
                max_results=self._clamp_items(max_results),
                only_posts_newer_than=only_posts_newer_than,
                only_posts_older_than=only_posts_older_than,
                timeout_secs=self.max_timeout_secs,
            )
        except (RuntimeError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)
