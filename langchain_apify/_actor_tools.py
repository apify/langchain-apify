"""Actor-specific tool subclasses.

Tools in this module wrap a single Apify Actor behind a simplified,
LLM-friendly interface. They inherit from
:class:`~langchain_apify.tools._ApifyGenericTool`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_apify._client import _DEFAULT_RUN_TIMEOUT_SECS
from langchain_apify.tools import (
    ApifyGoogleSearchInput,
    ApifyWebCrawlerInput,
    CrawlerType,
    _ApifyGenericTool,
    _serialize_tool_response,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun

# Default result/page limits per Actor tool. Kept here so the ``_run``
# signature defaults and the description strings stay in sync.
_DEFAULT_GOOGLE_SEARCH_MAX_RESULTS = 10
_DEFAULT_WEB_CRAWLER_MAX_PAGES = 10
_DEFAULT_WEB_CRAWLER_MAX_DEPTH = 1
_DEFAULT_WEB_CRAWLER_TYPE: CrawlerType = 'cheerio'
_DEFAULT_RAG_MAX_RESULTS = 5
_DEFAULT_GOOGLE_MAPS_MAX_RESULTS = 10
_DEFAULT_YOUTUBE_MAX_RESULTS = 10
_DEFAULT_ECOMMERCE_MAX_RESULTS = 20


def _item_metadata(item: dict) -> dict:
    """Return an item's ``metadata`` block, or ``{}`` if missing/non-dict.

    Some Actors surface a ``null`` (or otherwise non-dict) ``metadata`` value,
    so a plain ``item.get('metadata', {})`` would raise ``AttributeError`` on
    the chained ``.get(...)``.
    """
    meta = item.get('metadata')
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Search & Crawling tools
# ---------------------------------------------------------------------------


class ApifyGoogleSearchTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google and return structured results via Apify.

    Wraps the ``apify/google-search-scraper`` Actor behind a simplified,
    LLM-friendly interface.  Returns a normalized JSON envelope whose
    ``items`` are objects with ``title``, ``url``, and ``description`` keys.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run``, ``items`` (each
        ``{"title", "url", "description"}``), ``content``, and ``meta``.

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
        'Search Google using Apify and return a normalized JSON envelope.'
        ' Each item has keys: title, url, description.'
        ' Required: query (str) — the search query.'
        f' Optional: max_results (int, default {_DEFAULT_GOOGLE_SEARCH_MAX_RESULTS}),'
        ' country_code (str|null), language_code (str|null),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns keys: run, items, content, meta.'
    )
    args_schema: type[BaseModel] = ApifyGoogleSearchInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_GOOGLE_SEARCH_MAX_RESULTS,
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
        return _serialize_tool_response(tool_name=self.name, items=results)


class ApifyWebCrawlerTool(_ApifyGenericTool):  # type: ignore[override]
    """Crawl a website and return page content as JSON via Apify.

    Wraps the ``apify/website-content-crawler`` Actor.  Returns a normalized
    JSON envelope whose ``items`` are page objects with ``url``, ``title``,
    and ``content`` (markdown) keys.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run``, ``items`` (each
        ``{"url", "title", "content"}``), ``content``, and ``meta``.

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
        'Crawl a website using Apify and return a normalized JSON envelope.'
        ' Each item has keys: url, title, content.'
        ' Required: url (str) — seed URL to crawl.'
        f' Optional: max_crawl_pages (int, default {_DEFAULT_WEB_CRAWLER_MAX_PAGES}),'
        f' max_crawl_depth (int, default {_DEFAULT_WEB_CRAWLER_MAX_DEPTH}),'
        f' crawler_type (str, default "{_DEFAULT_WEB_CRAWLER_TYPE}"),'
        f' timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}).'
        ' Returns keys: run, items, content, meta.'
    )
    args_schema: type[BaseModel] = ApifyWebCrawlerInput

    def _run(
        self,
        url: str,
        max_crawl_pages: int = _DEFAULT_WEB_CRAWLER_MAX_PAGES,
        max_crawl_depth: int = _DEFAULT_WEB_CRAWLER_MAX_DEPTH,
        crawler_type: CrawlerType = _DEFAULT_WEB_CRAWLER_TYPE,
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
        pages = [
            {
                'url': item.get('url', ''),
                'title': _item_metadata(item).get('title', ''),
                'content': item.get('markdown') or item.get('text', ''),
            }
            for item in items
        ]
        return _serialize_tool_response(tool_name=self.name, items=pages)


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
    objects for RAG pipelines), this tool returns a normalized JSON envelope
    suitable for agent tool-calling.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run``, ``items`` (each
        ``{"url", "title", "content"}``), ``content``, and ``meta``.

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
        'Search the web and return a normalized JSON envelope with crawled results.'
        ' Each item has keys: url, title, content.'
        ' Required: query (str) - the search query.'
        f' Optional: max_results (int, default {_DEFAULT_RAG_MAX_RESULTS}).'
        ' Returns keys: run, items, content, meta.'
    )
    args_schema: type[BaseModel] = ApifyRAGWebBrowserInput

    def _run(
        self,
        query: str,
        max_results: int = _DEFAULT_RAG_MAX_RESULTS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.rag_web_browser_search(
                query,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        results = [
            {
                'url': _item_metadata(item).get('url') or item.get('crawledUrl', ''),
                'title': _item_metadata(item).get('title', ''),
                'content': item.get('markdown') or item.get('text', ''),
            }
            for item in items
        ]
        return _serialize_tool_response(
            tool_name=self.name,
            run=run,
            items=results,
        )


class ApifyGoogleMapsTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google Maps for places, reviews, and business details.

    Wraps the ``compass/crawler-google-places`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``), ``items`` (list of
        place dicts), ``content``, and ``meta``.

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
        'Search Google Maps places, reviews, and business details and return a normalized JSON envelope.'
        ' Required: query (str) - the search query.'
        f' Optional: max_results (int, default {_DEFAULT_GOOGLE_MAPS_MAX_RESULTS}),'
        ' language (str|null - ISO code, e.g. "en").'
        ' Returns keys: run, items, content, meta.'
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
        return _serialize_tool_response(
            tool_name=self.name,
            run=run,
            items=items,
        )


class ApifyYouTubeScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape YouTube videos, channels, or search results.

    Wraps the ``streamers/youtube-scraper`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``), ``items`` (list of
        video / channel dicts), ``content``, and ``meta``.

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
        'Scrape YouTube by keyword, video URL, or channel URL and return a normalized JSON envelope.'
        ' Required: search_query (str - keyword for "search" mode, or a video/channel URL).'
        ' Optional: search_type (one of "search", "video", "channel"; default "search"),'
        f' max_results (int, default {_DEFAULT_YOUTUBE_MAX_RESULTS}).'
        ' Returns keys: run, items, content, meta.'
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
        return _serialize_tool_response(
            tool_name=self.name,
            run=run,
            items=items,
        )


class ApifyEcommerceScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Extract product or listing data from an e-commerce URL.

    Wraps the ``apify/e-commerce-scraping-tool`` Actor.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``), ``items`` (list of
        product / listing dicts), ``content``, and ``meta``.

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
        'Extract product data from an e-commerce URL and return a normalized JSON envelope.'
        ' Required: url (str) - product-detail or category / listing URL.'
        ' Optional: url_type (one of "product", "category"; default "product"),'
        f' max_results (int, default {_DEFAULT_ECOMMERCE_MAX_RESULTS}).'
        ' Returns keys: run, items, content, meta.'
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
        return _serialize_tool_response(
            tool_name=self.name,
            run=run,
            items=items,
        )
