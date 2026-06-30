"""Search & crawling Actor tools and their input schemas.

Each tool wraps a single Apify search/crawl Actor behind a simplified,
LLM-friendly interface returning the standard JSON envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import ArgsSchema, ToolException
from pydantic import BaseModel, Field, field_validator

from langchain_apify._constants import (
    _DEFAULT_CRAWLER_TYPE,
    _DEFAULT_ECOMMERCE_MAX_RESULTS,
    _DEFAULT_GOOGLE_MAPS_MAX_RESULTS,
    _DEFAULT_GOOGLE_MAX_RESULTS,
    _DEFAULT_MAX_CRAWL_DEPTH,
    _DEFAULT_MAX_CRAWL_PAGES,
    _DEFAULT_RAG_MAX_RESULTS,
    _DEFAULT_RUN_TIMEOUT_SECS,
    _DEFAULT_YOUTUBE_MAX_RESULTS,
    _MAX_CRAWL_DEPTH_CAP,
    _MAX_ITEMS_CAP,
)
from langchain_apify._types import (  # noqa: TCH001  # runtime-needed: pydantic Field annotations
    CrawlerType,
    EcommerceUrlType,
    YouTubeSearchType,
)
from langchain_apify._utils import _extract_content, _extract_source, _safe_title
from langchain_apify.tools.base import _TOOL_RUN_ERRORS, _ApifyGenericTool
from langchain_apify.tools.core import _DESC_RUN_TIMEOUT_SECS

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class ApifyGoogleSearchInput(BaseModel):
    """Input schema for :class:`ApifyGoogleSearchTool`."""

    query: str = Field(description='Search query string.')
    max_results: int = Field(
        default=_DEFAULT_GOOGLE_MAX_RESULTS,
        description=f'Maximum number of search results to return (clamped to {_MAX_ITEMS_CAP} max).',
    )
    country_code: str | None = Field(
        default=None,
        description='Two-letter country code (case-insensitive; normalised to lowercase, e.g. "us", "gb").',
        pattern=r'^[a-zA-Z]{2}$',
    )
    language_code: str | None = Field(
        default=None,
        description='Two-letter language code (case-insensitive; normalised to lowercase, e.g. "en", "fr").',
        pattern=r'^[a-zA-Z]{2}$',
    )
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)

    @field_validator('country_code', 'language_code')
    @classmethod
    def _normalise_locale_code(cls, value: str | None) -> str | None:
        return value.lower() if value else value


class ApifyWebCrawlerInput(BaseModel):
    """Input schema for :class:`ApifyWebCrawlerTool`."""

    url: str = Field(description='Seed URL to start crawling from.')
    max_crawl_pages: int = Field(
        default=_DEFAULT_MAX_CRAWL_PAGES,
        description=f'Maximum number of pages to crawl (clamped to {_MAX_ITEMS_CAP} max).',
    )
    max_crawl_depth: int = Field(
        default=_DEFAULT_MAX_CRAWL_DEPTH,
        description=f'Maximum link-follow depth from the seed URL (clamped to {_MAX_CRAWL_DEPTH_CAP} max).',
    )
    crawler_type: CrawlerType = Field(
        default=_DEFAULT_CRAWLER_TYPE,
        description='Crawler engine: "cheerio" (fast, static HTML), "playwright:adaptive" or "playwright:firefox".',
    )
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)


class ApifyRAGWebBrowserInput(BaseModel):
    """Input schema for :class:`ApifyRAGWebBrowserTool`."""

    query: str = Field(description='Search query string.')
    max_results: int = Field(
        default=_DEFAULT_RAG_MAX_RESULTS,
        description=f'Maximum number of results to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


class ApifyGoogleMapsInput(BaseModel):
    """Input schema for :class:`ApifyGoogleMapsTool`."""

    query: str = Field(description='Search query (e.g. "coffee shops in Berlin").')
    max_results: int = Field(
        default=_DEFAULT_GOOGLE_MAPS_MAX_RESULTS,
        description=f'Maximum number of places to return (clamped to {_MAX_ITEMS_CAP} max).',
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
    search_type: YouTubeSearchType = Field(
        default='search',
        description='Scrape mode: search keyword, single video URL, or channel URL.',
    )
    max_results: int = Field(
        default=_DEFAULT_YOUTUBE_MAX_RESULTS,
        description=f'Maximum number of items to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


class ApifyEcommerceScraperInput(BaseModel):
    """Input schema for :class:`ApifyEcommerceScraperTool`."""

    url: str = Field(description='Product-detail URL or category / listing page URL to scrape.')
    url_type: EcommerceUrlType = Field(
        default='product',
        description=(
            'Type of page the URL points to: "product" for a product-detail page, '
            '"category" for a category / listing page.'
        ),
    )
    max_results: int = Field(
        default=_DEFAULT_ECOMMERCE_MAX_RESULTS,
        description=f'Maximum number of products to return (clamped to {_MAX_ITEMS_CAP} max).',
    )


# ---------------------------------------------------------------------------
# Tools
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
        ' Required: query (str); the search query.'
        f' Optional: max_results (int, default {_DEFAULT_GOOGLE_MAX_RESULTS}),'
        ' country_code (str|null), language_code (str|null),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyGoogleSearchInput

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
            run, results = self._client.google_search(
                query,
                max_results=self._clamp_items(max_results),
                country_code=country_code,
                language_code=language_code,
                timeout_secs=self._clamp_timeout(timeout_secs),
            )
        except _TOOL_RUN_ERRORS as exc:
            raise ToolException(str(exc)) from exc
        # default=str coerces any non-JSON-native types (e.g. datetime from
        # the Apify client's clean=True deserialiser) to their string repr
        # so the LLM never sees a serialisation failure.
        return self._envelope(run, results)


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
        ' Required: url (str); seed URL to crawl.'
        f' Optional: max_crawl_pages (int, default {_DEFAULT_MAX_CRAWL_PAGES}),'
        f' max_crawl_depth (int, default {_DEFAULT_MAX_CRAWL_DEPTH}),'
        f' crawler_type (str, default "{_DEFAULT_CRAWLER_TYPE}"),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyWebCrawlerInput

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
            run, items = self._client.crawl_website(
                url,
                max_crawl_pages=self._clamp_items(max_crawl_pages),
                max_crawl_depth=self._clamp_depth(max_crawl_depth),
                crawler_type=crawler_type,
                timeout_secs=self._clamp_timeout(timeout_secs),
            )
        except _TOOL_RUN_ERRORS as exc:
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
        return self._envelope(run, pages)


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
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyRAGWebBrowserInput

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
        except _TOOL_RUN_ERRORS as exc:
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
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyGoogleMapsInput

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
        except _TOOL_RUN_ERRORS as exc:
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
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyYouTubeScraperInput

    def _run(
        self,
        search_query: str,
        search_type: YouTubeSearchType = 'search',
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
        except _TOOL_RUN_ERRORS as exc:
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
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyEcommerceScraperInput

    def _run(
        self,
        url: str,
        url_type: EcommerceUrlType = 'product',
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
        except _TOOL_RUN_ERRORS as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, items)


# Convenience tool-class list for selective agent binding.
APIFY_SEARCH_TOOLS: list[type[BaseTool]] = [
    ApifyGoogleSearchTool,
    ApifyWebCrawlerTool,
    ApifyRAGWebBrowserTool,
    ApifyGoogleMapsTool,
    ApifyYouTubeScraperTool,
    ApifyEcommerceScraperTool,
]
