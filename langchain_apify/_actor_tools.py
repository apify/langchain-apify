"""Actor-specific tool subclasses.

Tools in this module wrap a single Apify Actor behind a simplified,
LLM-friendly interface. They inherit from
:class:`~langchain_apify.tools._ApifyGenericTool`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_apify.tools import (
    ApifyGoogleSearchInput,
    ApifyWebCrawlerInput,
    CrawlerType,
    _ApifyGenericTool,
    _run_meta,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun

# ---------------------------------------------------------------------------
# Search & Crawling tools
# ---------------------------------------------------------------------------


class ApifyGoogleSearchTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google and return structured results via Apify.

    Wraps the ``apify/google-search-scraper`` Actor behind a simplified,
    LLM-friendly interface.  Returns a JSON string containing an array of
    result objects, each with ``title``, ``url``, and ``description`` keys.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string — an array of ``{"title", "url", "description"}`` objects.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyGoogleSearchTool

            tool = ApifyGoogleSearchTool()
            results = tool.invoke({"query": "LangChain framework"})
    """

    name: str = 'apify_google_search'
    description: str = (
        'Search Google using Apify and return structured results as a JSON array.'
        ' Each result has keys: title, url, description.'
        ' Required: query (str) — the search query.'
        ' Optional: max_results (int, default 10),'
        ' country_code (str|null), language_code (str|null),'
        ' timeout_secs (int, default 300).'
    )
    args_schema: type[BaseModel] = ApifyGoogleSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        country_code: str | None = None,
        language_code: str | None = None,
        timeout_secs: int = 300,
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
        return json.dumps(results)


class ApifyWebCrawlerTool(_ApifyGenericTool):  # type: ignore[override]
    """Crawl a website and return page content as JSON via Apify.

    Wraps the ``apify/website-content-crawler`` Actor.  Returns a JSON string
    containing an array of page objects, each with ``url``, ``title``, and
    ``content`` (markdown) keys.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string — an array of ``{"url", "title", "content"}`` objects.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyWebCrawlerTool

            tool = ApifyWebCrawlerTool()
            pages = tool.invoke({
                "url": "https://docs.apify.com",
                "max_crawl_pages": 5,
            })
    """

    name: str = 'apify_web_crawler'
    description: str = (
        'Crawl a website using Apify and return page content as a JSON array.'
        ' Each page object has keys: url, title, content (markdown).'
        ' Required: url (str) — seed URL to crawl.'
        ' Optional: max_crawl_pages (int, default 10),'
        ' max_crawl_depth (int, default 1),'
        ' crawler_type (str, default "cheerio"),'
        ' timeout_secs (int, default 300).'
    )
    args_schema: type[BaseModel] = ApifyWebCrawlerInput

    def _run(
        self,
        url: str,
        max_crawl_pages: int = 10,
        max_crawl_depth: int = 1,
        crawler_type: CrawlerType = 'cheerio',
        timeout_secs: int = 300,
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
                'title': item.get('metadata', {}).get('title', ''),
                'content': item.get('markdown') or item.get('text', ''),
            }
            for item in items
        ]
        return json.dumps(pages)


# ---------------------------------------------------------------------------
# Input schemas (US-4 Search & Crawling Actor tools)
# ---------------------------------------------------------------------------


class ApifyRAGWebBrowserInput(BaseModel):
    """Input schema for :class:`ApifyRAGWebBrowserTool`."""

    query: str = Field(description='Search query string.')
    max_results: int = Field(default=5, description='Maximum number of results to return.')


class ApifyGoogleMapsInput(BaseModel):
    """Input schema for :class:`ApifyGoogleMapsTool`."""

    query: str = Field(description='Search query (e.g. "coffee shops in Berlin").')
    max_results: int = Field(default=10, description='Maximum number of places to return.')
    language: str | None = Field(
        default=None,
        description='Optional ISO language code for results (e.g. "en", "de").',
    )


class ApifyYouTubeScraperInput(BaseModel):
    """Input schema for :class:`ApifyYouTubeScraperTool`."""

    search_query: str = Field(
        description=(
            'Keyword for "search" mode, or a video/channel URL for "video"/"channel" modes.'
        ),
    )
    search_type: Literal['search', 'video', 'channel'] = Field(
        default='search',
        description=(
            'Type of scrape: "search" for a keyword search, "video" for a video URL, '
            '"channel" for a channel URL.'
        ),
    )
    max_results: int = Field(default=10, description='Maximum number of items to return.')


class ApifyEcommerceScraperInput(BaseModel):
    """Input schema for :class:`ApifyEcommerceScraperTool`."""

    url: str = Field(description='Product, category, or listing URL to scrape.')
    max_results: int = Field(default=20, description='Maximum number of items to return.')


# ---------------------------------------------------------------------------
# Tools (US-4 Search & Crawling Actor tools)
# ---------------------------------------------------------------------------


class ApifyRAGWebBrowserTool(_ApifyGenericTool):  # type: ignore[override]
    """Search the web and return content from top results.

    Wraps the ``apify/rag-web-browser`` Actor.  Unlike
    :class:`ApifySearchRetriever` (which returns LangChain ``Document``
    objects for RAG pipelines), this tool returns a JSON string suitable
    for agent tool-calling.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of crawled-page dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyRAGWebBrowserTool

            tool = ApifyRAGWebBrowserTool()
            result = tool.invoke({"query": "what is LangChain?", "max_results": 3})
    """

    name: str = 'apify_rag_web_browser'
    description: str = (
        'Search the web and return content from the top results as JSON.'
        ' Required: query (str) - the search query.'
        ' Optional: max_results (int, default 5).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyRAGWebBrowserInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
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
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyGoogleMapsTool(_ApifyGenericTool):  # type: ignore[override]
    """Search Google Maps for places, reviews, and business details.

    Wraps the ``compass/crawler-google-places`` Actor.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of place dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyGoogleMapsTool

            tool = ApifyGoogleMapsTool()
            result = tool.invoke({"query": "coffee shops in Berlin", "max_results": 5})
    """

    name: str = 'apify_google_maps'
    description: str = (
        'Search Google Maps places, reviews, and business details and return them as JSON.'
        ' Required: query (str) - the search query.'
        ' Optional: max_results (int, default 10), language (str|null - ISO code, e.g. "en").'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyGoogleMapsInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
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
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyYouTubeScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape YouTube videos, channels, or search results.

    Wraps the ``streamers/youtube-scraper`` Actor.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of video / channel dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

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
        'Scrape YouTube by keyword, video URL, or channel URL and return the results as JSON.'
        ' Required: search_query (str - keyword for "search" mode, or a video/channel URL).'
        ' Optional: search_type (one of "search", "video", "channel"; default "search"),'
        ' max_results (int, default 10).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyYouTubeScraperInput

    def _run(
        self,
        search_query: str,
        search_type: Literal['search', 'video', 'channel'] = 'search',
        max_results: int = 10,
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
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyEcommerceScraperTool(_ApifyGenericTool):  # type: ignore[override]
    """Extract product or listing data from an e-commerce URL.

    Wraps the ``apify/e-commerce-scraping-tool`` Actor.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of product / listing dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyEcommerceScraperTool

            tool = ApifyEcommerceScraperTool()
            result = tool.invoke({
                "url": "https://shop.example.com/category/123",
                "max_results": 20,
            })
    """

    name: str = 'apify_ecommerce_scraper'
    description: str = (
        'Extract product or listing data from an e-commerce URL and return it as JSON.'
        ' Required: url (str) - the product, category, or listing URL.'
        ' Optional: max_results (int, default 20).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at) and items.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: type[BaseModel] = ApifyEcommerceScraperInput

    def _run(
        self,
        url: str,
        max_results: int = 20,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.ecommerce_scrape(
                url,
                max_results=self._clamp_items(max_results),
                timeout_secs=self.max_timeout_secs,
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})
