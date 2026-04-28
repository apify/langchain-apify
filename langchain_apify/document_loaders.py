from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_apify._client import ApifyToolsClient
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import _create_apify_client

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_apify.tools import CrawlerType


class ApifyDatasetLoader(BaseLoader, BaseModel):
    """Load datasets from Apify web scraping, crawling, and data extraction platform.

    To use, you should have the environment variable `APIFY_API_TOKEN` set
    with your API key, or pass `apify_api_token`
    as a named parameter to the constructor.

    For details, see https://docs.apify.com/platform/integrations/langchain

    Example:
        .. code-block:: python

            from langchain_apify import ApifyDatasetLoader
            from langchain_core.documents import Document

            loader = ApifyDatasetLoader(
                dataset_id="YOUR-DATASET-ID",
                dataset_mapping_function=lambda dataset_item: Document(
                    page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
                ),
            )
            documents = loader.load()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    apify_api_token: SecretStr | None = Field(
        default_factory=secret_from_env('APIFY_API_TOKEN', default=None),
        description='Apify API token. Falls back to APIFY_API_TOKEN / APIFY_TOKEN environment variables.',
        exclude=True,
        repr=False,
    )
    apify_client: ApifyClient = Field(default=None, exclude=True)  # type: ignore[assignment]
    dataset_id: str
    """The ID of the dataset on the Apify platform."""
    dataset_mapping_function: Callable[[dict], Document]
    """A custom function that takes a single dictionary (an Apify dataset item)
     and converts it to an instance of the Document class."""

    def __init__(
        self,
        dataset_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        apify_api_token: str | SecretStr | None = None,
    ) -> None:
        """Initialize the loader with an Apify dataset ID and a mapping function.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an instance
                of the Document class.
            apify_api_token (str | SecretStr): Apify API token. Falls back to the
                ``APIFY_API_TOKEN`` / ``APIFY_TOKEN`` environment variables.
        """
        init_kwargs: dict[str, Any] = {
            'dataset_id': dataset_id,
            'dataset_mapping_function': dataset_mapping_function,
        }
        # Only forward the token when explicitly provided; otherwise let the
        # Pydantic ``default_factory`` read it from the environment.
        if apify_api_token is not None:
            init_kwargs['apify_api_token'] = apify_api_token
        super().__init__(**init_kwargs)

    @model_validator(mode='after')
    def _init_client(self) -> ApifyDatasetLoader:
        """Resolve the Apify API token and initialise the client.

        Checks ``APIFY_TOKEN`` as a secondary fallback for code running on the
        Apify platform where only that variable is set.

        Returns:
            ApifyDatasetLoader: The validated loader instance.

        Raises:
            ValueError: If no token is available from any source.
        """
        token = self.apify_api_token
        if token is None:
            # Secondary fallback for code running on the Apify platform.
            raw = os.getenv('APIFY_TOKEN')
            if raw:
                token = SecretStr(raw)
        if token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self.apify_client = _create_apify_client(ApifyClient, token.get_secret_value())
        return self

    def load(self) -> list[Document]:
        """Load documents.

        Returns:
            list[Document]: A list of mapped Document objects.
        """
        dataset_items = self.apify_client.dataset(self.dataset_id).list_items(clean=True).items
        return list(map(self.dataset_mapping_function, dataset_items))

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents.

        Yields:
            Document: A mapped Document object.
        """
        dataset_items = self.apify_client.dataset(self.dataset_id).iterate_items(
            clean=True,
        )
        for item in dataset_items:
            yield self.dataset_mapping_function(item)


class ApifyCrawlLoader(BaseLoader):
    """Crawl a website and load pages as LangChain Documents.

    Wraps the ``apify/website-content-crawler`` Actor.  Runs a crawl starting
    from the seed URL and converts each crawled page into a ``Document`` with
    markdown content and metadata (source URL, title, crawl depth).

    Args:
        url: Seed URL to start crawling from.
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.
        max_crawl_pages: Maximum number of pages to crawl.
        max_crawl_depth: Maximum link-follow depth from the seed URL.
        crawler_type: Crawler engine (e.g. ``"cheerio"``, ``"playwright"``).
        timeout_secs: Maximum time in seconds to wait for the crawl.

    Returns:
        Iterator (or list) of ``Document`` objects.  ``page_content`` contains
        the page markdown; ``metadata`` includes ``source``, ``title``, and
        ``crawl_depth``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyCrawlLoader

            loader = ApifyCrawlLoader(
                url="https://docs.apify.com",
                max_crawl_pages=5,
            )
            documents = loader.load()
    """

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        apify_api_token: str | SecretStr | None = None,
        *,
        max_crawl_pages: int = 10,
        max_crawl_depth: int = 1,
        crawler_type: CrawlerType = 'cheerio',
        timeout_secs: int = 300,
    ) -> None:
        self.url = url
        self.max_crawl_pages = max_crawl_pages
        self.max_crawl_depth = max_crawl_depth
        self.crawler_type = crawler_type
        self.timeout_secs = timeout_secs
        self._client = ApifyToolsClient(apify_api_token=apify_api_token)

    def lazy_load(self) -> Iterator[Document]:
        """Crawl the website and yield Documents lazily.

        Yields:
            Document: One document per crawled page.
        """
        items = self._client.crawl_website(
            self.url,
            max_crawl_pages=self.max_crawl_pages,
            max_crawl_depth=self.max_crawl_depth,
            crawler_type=self.crawler_type,
            timeout_secs=self.timeout_secs,
        )
        for item in items:
            page_content = item.get('markdown') or item.get('text') or ''
            metadata: dict[str, Any] = {
                'source': item.get('url', ''),
                'title': item.get('metadata', {}).get('title', '') if isinstance(item.get('metadata'), dict) else '',
                'crawl_depth': item.get('crawlDepth', 0),
            }
            yield Document(page_content=page_content, metadata=metadata)
