from __future__ import annotations

import os

import httpx
from apify_client import ApifyClient
from apify_client.errors import ApifyClientError
from pydantic import SecretStr

from langchain_apify._error_messages import (
    _ERROR_ACTOR_RUN_FAILED,
    _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET,
    _ERROR_SCRAPE_EMPTY,
)
from langchain_apify._utils import _create_apify_client

# Only catches ApifyClientError and httpx.HTTPError. Other errors propagate.
_TRANSPORT_EXCEPTIONS = (ApifyClientError, httpx.HTTPError)

_SCRAPE_ACTOR_ID = 'apify/website-content-crawler'
_CRAWL_ACTOR_ID = 'apify/website-content-crawler'
_GOOGLE_SEARCH_ACTOR_ID = 'apify/google-search-scraper'
_RAG_WEB_BROWSER_ACTOR_ID = 'apify/rag-web-browser'
_GOOGLE_MAPS_ACTOR_ID = 'compass/crawler-google-places'
_YOUTUBE_SCRAPER_ACTOR_ID = 'streamers/youtube-scraper'
_ECOMMERCE_SCRAPER_ACTOR_ID = 'apify/e-commerce-scraping-tool'

_YOUTUBE_SEARCH_TYPES = ('search', 'video', 'channel')
_ECOMMERCE_URL_TYPES = ('product', 'category')
_DEFAULT_RUN_TIMEOUT_SECS = 300
_DEFAULT_SCRAPE_TIMEOUT_SECS = 120
_DEFAULT_CRAWL_TIMEOUT_SECS = 300
_DEFAULT_DATASET_ITEMS_LIMIT = 100
_RUN_STATUS_SUCCEEDED = 'SUCCEEDED'


class ApifyToolsClient:
    """Internal helper that wraps ``ApifyClient`` for the tools layer.

    One convenience method per tool operation. All methods are synchronous and
    block until the Actor run finishes.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Raises:
        ValueError: If no token is provided and the env var is not set.
    """

    def __init__(self, apify_api_token: SecretStr | str | None = None) -> None:
        if isinstance(apify_api_token, SecretStr):
            _token: str | None = apify_api_token.get_secret_value()
        else:
            _token = apify_api_token or os.getenv('APIFY_API_TOKEN')

        if not _token:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._client = _create_apify_client(ApifyClient, _token)

    def run_actor(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
    ) -> dict:
        """Start an Actor and block until it finishes.

        Args:
            actor_id: Actor ID or name (e.g. ``"apify/python-example"``).
            run_input: JSON-serialisable input for the Actor.
            timeout_secs: Maximum time to wait for the run to finish.
            memory_mbytes: Memory limit for the run, or *None* for Actor default.

        Returns:
            Full run-details dict returned by the Apify API.

        Raises:
            RuntimeError: If the run does not finish with status ``SUCCEEDED``.
        """
        call_kwargs: dict = {'run_input': run_input, 'timeout_secs': timeout_secs, 'logger': None}
        if memory_mbytes is not None:
            call_kwargs['memory_mbytes'] = memory_mbytes

        try:
            run = self._client.actor(actor_id).call(**call_kwargs)
        except _TRANSPORT_EXCEPTIONS as exc:
            msg = f'Apify Actor call failed for {actor_id}: {exc}'
            raise RuntimeError(msg) from exc
        if run is None:
            msg = f'Actor {actor_id} call returned no run details.'
            raise RuntimeError(msg)
        self._check_run_status(run)
        return run

    def get_dataset_items(
        self, dataset_id: str, limit: int = _DEFAULT_DATASET_ITEMS_LIMIT, offset: int = 0
    ) -> list[dict]:
        """Fetch items from an existing dataset.

        Args:
            dataset_id: Apify dataset ID.
            limit: Maximum number of items to return.
            offset: Number of items to skip from the start.

        Returns:
            List of dataset item dicts (may be empty).
        """
        try:
            return self._client.dataset(dataset_id).list_items(limit=limit, offset=offset, clean=True).items
        except _TRANSPORT_EXCEPTIONS as exc:
            msg = f'Apify dataset fetch failed for {dataset_id}: {exc}'
            raise RuntimeError(msg) from exc

    def run_actor_and_get_items(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = _DEFAULT_DATASET_ITEMS_LIMIT,
    ) -> tuple[dict, list[dict]]:
        """Run an Actor, then fetch items from its default dataset.

        Args:
            actor_id: Actor ID or name.
            run_input: JSON-serialisable input for the Actor.
            timeout_secs: Maximum time to wait for the run to finish.
            memory_mbytes: Memory limit for the run, or *None* for Actor default.
            dataset_items_limit: Maximum number of dataset items to return.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            RuntimeError: If the run does not finish with status ``SUCCEEDED``.
        """
        run = self.run_actor(actor_id, run_input, timeout_secs, memory_mbytes)
        dataset_id = run.get('defaultDatasetId')
        if not dataset_id:
            msg = f'Actor {actor_id} run succeeded but returned no default dataset ID.'
            raise RuntimeError(msg)
        items = self._list_items_or_raise(dataset_id, dataset_items_limit)
        return run, items

    def run_task(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
    ) -> dict:
        """Start a saved Actor task and block until it finishes.

        Args:
            task_id: Task ID or name (e.g. ``"user/my-task"``).
            task_input: JSON-serialisable input that overrides the task's
                pre-saved input.
            timeout_secs: Maximum time to wait for the run to finish.
            memory_mbytes: Memory limit for the run, or *None* for task default.

        Returns:
            Full run-details dict returned by the Apify API.

        Raises:
            RuntimeError: If the run does not finish with status ``SUCCEEDED``.
        """
        call_kwargs: dict = {'task_input': task_input, 'timeout_secs': timeout_secs}
        if memory_mbytes is not None:
            call_kwargs['memory_mbytes'] = memory_mbytes

        try:
            run = self._client.task(task_id).call(**call_kwargs)
        except _TRANSPORT_EXCEPTIONS as exc:
            msg = f'Apify task call failed for {task_id}: {exc}'
            raise RuntimeError(msg) from exc
        if run is None:
            msg = f'Task {task_id} call returned no run details.'
            raise RuntimeError(msg)
        self._check_run_status(run)
        return run

    def run_task_and_get_items(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = _DEFAULT_DATASET_ITEMS_LIMIT,
    ) -> tuple[dict, list[dict]]:
        """Run a saved Actor task, then fetch items from its default dataset.

        Args:
            task_id: Task ID or name.
            task_input: JSON-serialisable input that overrides the task's
                pre-saved input.
            timeout_secs: Maximum time to wait for the run to finish.
            memory_mbytes: Memory limit for the run, or *None* for task default.
            dataset_items_limit: Maximum number of dataset items to return.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            RuntimeError: If the run does not finish with status ``SUCCEEDED``.
        """
        run = self.run_task(task_id, task_input, timeout_secs, memory_mbytes)
        dataset_id = run.get('defaultDatasetId')
        if not dataset_id:
            msg = f'Task {task_id} run succeeded but returned no default dataset ID.'
            raise RuntimeError(msg)
        items = self._list_items_or_raise(dataset_id, dataset_items_limit)
        return run, items

    def scrape_url(self, url: str, timeout_secs: int = _DEFAULT_SCRAPE_TIMEOUT_SECS) -> str:
        """Scrape a single URL and return its content as markdown.

        Uses ``apify/website-content-crawler`` with ``maxCrawlPages=1``.

        Args:
            url: The URL to scrape.
            timeout_secs: Maximum time to wait for the crawl to finish.

        Returns:
            Markdown (or plain-text fallback) content of the page.

        Raises:
            RuntimeError: If the Actor run fails or no content is extracted.
        """
        run_input = {
            'startUrls': [{'url': url}],
            'maxCrawlPages': 1,
        }
        _, items = self.run_actor_and_get_items(
            _SCRAPE_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=1,
        )
        if not items:
            msg = _ERROR_SCRAPE_EMPTY.format(url=url)
            raise RuntimeError(msg)

        content = items[0].get('markdown') or items[0].get('text') or ''
        if not content:
            msg = _ERROR_SCRAPE_EMPTY.format(url=url)
            raise RuntimeError(msg)
        return content

    def google_search(
        self,
        query: str,
        max_results: int = 10,
        country_code: str | None = None,
        language_code: str | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> list[dict]:
        """Run a Google search and return structured results.

        Uses ``apify/google-search-scraper`` with a single query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            country_code: Two-letter country code for localised results.
            language_code: Two-letter language code.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            List of result dicts, each with ``title``, ``url``, and
            ``description`` keys.

        Raises:
            RuntimeError: If the Actor run fails.
        """
        run_input: dict = {
            'queries': query,
            'maxPagesPerQuery': 1,
            'resultsPerPage': max_results,
        }
        if country_code is not None:
            run_input['countryCode'] = country_code
        if language_code is not None:
            run_input['languageCode'] = language_code

        _, items = self.run_actor_and_get_items(
            _GOOGLE_SEARCH_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )
        results: list[dict] = [
            {
                'title': organic.get('title', ''),
                'url': organic.get('url', ''),
                'description': organic.get('description', ''),
            }
            for item in items
            for organic in item.get('organicResults', [])
        ]
        return results[:max_results]

    def rag_web_browser_search(
        self,
        query: str,
        max_results: int = 5,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Search the web and return crawled page content for RAG.

        Uses ``apify/rag-web-browser``.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple. Each item dict has at least
            ``crawledUrl``, ``text``, and a nested ``metadata`` block (among
            other keys returned by the Actor).

        Raises:
            RuntimeError: If the Actor run fails.
        """
        run_input: dict = {
            'query': query,
            'maxResults': max_results,
        }
        return self.run_actor_and_get_items(
            _RAG_WEB_BROWSER_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def google_maps_search(
        self,
        query: str,
        max_results: int = 10,
        language: str | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Search Google Maps places, reviews, and business details.

        Uses ``compass/crawler-google-places``.

        Args:
            query: Search query string (e.g. ``"coffee shops in Berlin"``).
            max_results: Maximum number of places to return.
            language: Optional ISO language code for results (e.g. ``"en"``).
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple where each item is a place dict.

        Raises:
            RuntimeError: If the Actor run fails.
        """
        run_input: dict = {
            'searchStringsArray': [query],
            'maxCrawledPlacesPerSearch': max_results,
        }
        if language is not None:
            run_input['language'] = language

        return self.run_actor_and_get_items(
            _GOOGLE_MAPS_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def youtube_scrape(
        self,
        search_query: str,
        search_type: str = 'search',
        max_results: int = 10,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape YouTube videos, channels, or search results.

        Uses ``streamers/youtube-scraper``.

        Args:
            search_query: Keyword for ``search`` mode, or a video/channel URL
                for ``video``/``channel`` modes.
            search_type: One of ``"search"``, ``"video"``, ``"channel"``.
            max_results: Maximum number of items to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            ValueError: If ``search_type`` is not a supported value.
            RuntimeError: If the Actor run fails.
        """
        if search_type not in _YOUTUBE_SEARCH_TYPES:
            msg = f'Invalid search_type {search_type!r}; expected one of {_YOUTUBE_SEARCH_TYPES}.'
            raise ValueError(msg)

        run_input: dict = {'maxResults': max_results}
        if search_type == 'search':
            run_input['searchKeywords'] = search_query
        else:
            run_input['startUrls'] = [{'url': search_query}]

        return self.run_actor_and_get_items(
            _YOUTUBE_SCRAPER_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def ecommerce_scrape(
        self,
        url: str,
        url_type: str = 'product',
        max_results: int = 20,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Extract product data from an e-commerce URL.

        Uses ``apify/e-commerce-scraping-tool``. ``url_type`` selects which
        Actor input field the URL is sent as: ``"product"`` → ``detailsUrls``
        (a single product-detail page), ``"category"`` → ``listingUrls``
        (a category / listing page that the Actor will expand into product
        results).

        Args:
            url: Product-detail or category / listing URL to scrape.
            url_type: One of ``"product"`` or ``"category"``.
            max_results: Maximum number of products to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            ValueError: If ``url_type`` is not a supported value.
            RuntimeError: If the Actor run fails.
        """
        if url_type not in _ECOMMERCE_URL_TYPES:
            msg = f'Invalid url_type {url_type!r}; expected one of {_ECOMMERCE_URL_TYPES}.'
            raise ValueError(msg)

        input_key = 'detailsUrls' if url_type == 'product' else 'listingUrls'
        run_input: dict = {
            input_key: [{'url': url}],
            'maxProductResults': max_results,
        }
        return self.run_actor_and_get_items(
            _ECOMMERCE_SCRAPER_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def crawl_website(
        self,
        url: str,
        max_crawl_pages: int = 10,
        max_crawl_depth: int = 1,
        crawler_type: str = 'cheerio',
        timeout_secs: int = _DEFAULT_CRAWL_TIMEOUT_SECS,
    ) -> list[dict]:
        """Crawl a website and return page content.

        Uses ``apify/website-content-crawler``.

        Args:
            url: Seed URL to start crawling from.
            max_crawl_pages: Maximum number of pages to crawl.
            max_crawl_depth: Maximum link-follow depth from the seed URL.
            crawler_type: Crawler engine (e.g. ``"cheerio"``, ``"playwright"``).
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            List of page dicts, each with at least ``url``, ``title``, and
            ``markdown`` (or ``text``) keys.

        Raises:
            RuntimeError: If the Actor run fails.
        """
        run_input: dict = {
            'startUrls': [{'url': url}],
            'maxCrawlPages': max_crawl_pages,
            'maxCrawlDepth': max_crawl_depth,
            'crawlerType': crawler_type,
        }
        _, items = self.run_actor_and_get_items(
            _CRAWL_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_crawl_pages,
        )
        return items

    def _list_items_or_raise(self, dataset_id: str, limit: int) -> list[dict]:
        """Fetch dataset items, wrapping any network error in a RuntimeError."""
        try:
            return self._client.dataset(dataset_id).list_items(limit=limit, clean=True).items
        except _TRANSPORT_EXCEPTIONS as exc:
            msg = f'Apify dataset fetch failed for {dataset_id}: {exc}'
            raise RuntimeError(msg) from exc

    @staticmethod
    def _check_run_status(run: dict) -> None:
        """Raise if the run did not succeed."""
        status = run.get('status')
        if status != _RUN_STATUS_SUCCEEDED:
            run_id = run.get('id', 'unknown')
            msg = _ERROR_ACTOR_RUN_FAILED.format(run_id=run_id, status=status)
            raise RuntimeError(msg)
