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
_INSTAGRAM_ACTOR_ID = 'apify/instagram-scraper'
_LINKEDIN_POSTS_ACTOR_ID = 'apimaestro/linkedin-profile-posts'
_LINKEDIN_SEARCH_ACTOR_ID = 'harvestapi/linkedin-profile-search'
_LINKEDIN_DETAIL_ACTOR_ID = 'apimaestro/linkedin-profile-detail'
_TWITTER_ACTOR_ID = 'apidojo/twitter-scraper-lite'
_TIKTOK_ACTOR_ID = 'clockworks/tiktok-scraper'
_FACEBOOK_ACTOR_ID = 'apify/facebook-posts-scraper'
_DEFAULT_RUN_TIMEOUT_SECS = 300
_DEFAULT_SCRAPE_TIMEOUT_SECS = 120
_DEFAULT_SOCIAL_TIMEOUT_SECS = 600
_DEFAULT_DATASET_ITEMS_LIMIT = 100
_DEFAULT_SOCIAL_RESULTS_LIMIT = 20
_RUN_STATUS_SUCCEEDED = 'SUCCEEDED'

# Instagram-specific mappings
_INSTAGRAM_RESULTS_TYPE_MAP = {
    'user': 'posts',
    'hashtag': 'posts',
    'post': 'posts',
    'comments': 'comments',
}


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

    def instagram_scrape(
        self,
        search_type: str,
        search_query: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        only_posts_newer_than: str | None = None,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape Instagram via ``apify/instagram-scraper``.

        Args:
            search_type: One of ``"user"``, ``"hashtag"``, ``"post"``, ``"comments"``.
            search_query: Username, hashtag, or Instagram URL depending on
                ``search_type``.
            max_results: Maximum number of items to return.
            only_posts_newer_than: Optional date filter. Accepts ``YYYY-MM-DD``,
                ISO-8601, or relative (e.g. ``"1 day"``, ``"2 months"``).
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            ValueError: If ``search_type`` is not recognised.
            RuntimeError: If the Actor run does not succeed.
        """
        results_type = _INSTAGRAM_RESULTS_TYPE_MAP.get(search_type)
        if results_type is None:
            msg = (
                f'Unsupported Instagram search_type {search_type!r}. '
                f'Expected one of: {sorted(_INSTAGRAM_RESULTS_TYPE_MAP)}.'
            )
            raise ValueError(msg)

        direct_url = self._build_instagram_url(search_type, search_query)
        run_input: dict = {
            'directUrls': [direct_url],
            'resultsType': results_type,
            'resultsLimit': max_results,
        }
        if only_posts_newer_than is not None:
            run_input['onlyPostsNewerThan'] = only_posts_newer_than
        return self.run_actor_and_get_items(
            _INSTAGRAM_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def linkedin_profile_posts(
        self,
        profile_url: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape LinkedIn profile posts via ``apimaestro/linkedin-profile-posts``.

        Args:
            profile_url: LinkedIn profile URL or username.
            max_results: Maximum number of posts to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {
            'username': profile_url,
            'total_posts': max_results,
        }
        return self.run_actor_and_get_items(
            _LINKEDIN_POSTS_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def linkedin_profile_search(
        self,
        query: str,
        max_results: int = 10,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Search LinkedIn profiles via ``harvestapi/linkedin-profile-search``.

        Args:
            query: Search keywords (e.g., name, title, company).
            max_results: Maximum number of profiles to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {
            'searchQuery': query,
            'maxItems': max_results,
        }
        return self.run_actor_and_get_items(
            _LINKEDIN_SEARCH_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def linkedin_profile_detail(
        self,
        profile_url: str,
        *,
        include_email: bool = False,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Fetch a LinkedIn profile via ``apimaestro/linkedin-profile-detail``.

        Args:
            profile_url: LinkedIn profile URL or username.
            include_email: If True, attempt to include the profile email when
                available.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple. ``items`` typically contains a
            single profile dict.

        Raises:
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {
            'username': profile_url,
            'includeEmail': include_email,
        }
        return self.run_actor_and_get_items(
            _LINKEDIN_DETAIL_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=1,
        )

    def twitter_scrape(  # noqa: PLR0913
        self,
        search_query: str,
        search_mode: str = 'search',
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        start: str | None = None,
        end: str | None = None,
        sort: str | None = None,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape Twitter/X via ``apidojo/twitter-scraper-lite``.

        Args:
            search_query: Search term, username, or tweet URL.
            search_mode: One of ``"search"``, ``"user"``, ``"replies"``.
            max_results: Maximum number of tweets to return.
            start: Optional ISO-8601 start date — only return tweets newer
                than this date.
            end: Optional ISO-8601 end date — only return tweets older than
                this date.
            sort: Optional sort order. One of ``"Latest"`` or ``"Top"``.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            ValueError: If ``search_mode`` is not recognised.
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {'maxItems': max_results}
        if search_mode == 'search':
            run_input['searchTerms'] = [search_query]
        elif search_mode == 'user':
            run_input['twitterHandles'] = [search_query.lstrip('@')]
        elif search_mode == 'replies':
            run_input['startUrls'] = [search_query]
        else:
            msg = f"Unsupported Twitter search_mode {search_mode!r}. Expected one of: ['search', 'user', 'replies']."
            raise ValueError(msg)
        if start is not None:
            run_input['start'] = start
        if end is not None:
            run_input['end'] = end
        if sort is not None:
            run_input['sort'] = sort
        return self.run_actor_and_get_items(
            _TWITTER_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def tiktok_scrape(
        self,
        search_query: str,
        search_type: str = 'search',
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape TikTok via ``clockworks/tiktok-scraper``.

        Args:
            search_query: Username, hashtag, or search keyword.
            search_type: One of ``"search"``, ``"user"``, ``"hashtag"``.
            max_results: Maximum number of items to return.
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            ValueError: If ``search_type`` is not recognised.
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {'resultsPerPage': max_results}
        if search_type == 'search':
            run_input['searchQueries'] = [search_query]
        elif search_type == 'user':
            run_input['profiles'] = [search_query.lstrip('@')]
        elif search_type == 'hashtag':
            run_input['hashtags'] = [search_query.lstrip('#')]
        else:
            msg = f"Unsupported TikTok search_type {search_type!r}. Expected one of: ['search', 'user', 'hashtag']."
            raise ValueError(msg)
        return self.run_actor_and_get_items(
            _TIKTOK_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    def facebook_posts_scrape(
        self,
        page_url: str,
        max_results: int = _DEFAULT_SOCIAL_RESULTS_LIMIT,
        only_posts_newer_than: str | None = None,
        only_posts_older_than: str | None = None,
        timeout_secs: int = _DEFAULT_SOCIAL_TIMEOUT_SECS,
    ) -> tuple[dict, list[dict]]:
        """Scrape Facebook page posts via ``apify/facebook-posts-scraper``.

        Args:
            page_url: Facebook page URL.
            max_results: Maximum number of posts to return.
            only_posts_newer_than: Optional date filter. Accepts ``YYYY-MM-DD``,
                ISO-8601, or relative (e.g. ``"1 day"``, ``"2 months"``).
            only_posts_older_than: Optional date filter. Accepts ``YYYY-MM-DD``,
                ISO-8601, or relative (e.g. ``"1 day"``, ``"2 months"``).
            timeout_secs: Maximum time to wait for the run to finish.

        Returns:
            A ``(run_details, items)`` tuple.

        Raises:
            RuntimeError: If the Actor run does not succeed.
        """
        run_input: dict = {
            'startUrls': [{'url': page_url}],
            'resultsLimit': max_results,
        }
        if only_posts_newer_than is not None:
            run_input['onlyPostsNewerThan'] = only_posts_newer_than
        if only_posts_older_than is not None:
            run_input['onlyPostsOlderThan'] = only_posts_older_than
        return self.run_actor_and_get_items(
            _FACEBOOK_ACTOR_ID,
            run_input=run_input,
            timeout_secs=timeout_secs,
            dataset_items_limit=max_results,
        )

    @staticmethod
    def _build_instagram_url(search_type: str, search_query: str) -> str:
        """Build an Instagram URL from a username/hashtag/URL based on search type."""
        if search_query.startswith(('http://', 'https://')):
            return search_query
        if search_type == 'hashtag':
            tag = search_query.lstrip('#')
            return f'https://www.instagram.com/explore/tags/{tag}/'
        if search_type == 'user':
            handle = search_query.lstrip('@')
            return f'https://www.instagram.com/{handle}/'
        # post/comments expect a URL; if a bare ID is given, build a /p/ URL
        return f'https://www.instagram.com/p/{search_query}/'

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
            if status_message := run.get('statusMessage'):
                msg = f'{msg} {status_message}'
            raise RuntimeError(msg)
