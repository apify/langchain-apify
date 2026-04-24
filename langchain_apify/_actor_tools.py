"""Actor-specific tool subclasses (search, social-media, etc.).

Downstream feature branches add concrete tools here.  They inherit from
:class:`~langchain_apify.tools._ApifyGenericTool` and use
:func:`~langchain_apify.tools._run_meta` to format run metadata.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import ToolException

from langchain_apify.tools import (
    ApifyGoogleSearchInput,
    ApifyWebCrawlerInput,
    _ApifyGenericTool,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel

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
        ' country_code (str|null), language_code (str|null).'
    )
    args_schema: type[BaseModel] = ApifyGoogleSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        country_code: str | None = None,
        language_code: str | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            results = self._client.google_search(
                query,
                max_results=self._clamp_items(max_results),
                country_code=country_code,
                language_code=language_code,
                timeout_secs=self.max_timeout_secs,
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
        crawler_type: str = 'cheerio',
        timeout_secs: int = 300,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            items = self._client.crawl_website(
                url,
                max_crawl_pages=self._clamp_items(max_crawl_pages),
                max_crawl_depth=max_crawl_depth,
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
# Social-media tools
# ---------------------------------------------------------------------------
