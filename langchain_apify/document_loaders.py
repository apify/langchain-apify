from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from langchain_apify._client import ApifyToolsClient
from langchain_apify._constants import (
    _DEFAULT_CRAWLER_TYPE,
    _DEFAULT_MAX_CRAWL_DEPTH,
    _DEFAULT_MAX_CRAWL_PAGES,
    _DEFAULT_RUN_TIMEOUT_SECS,
)
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _apify_token_secret_factory,
    _create_apify_client,
    _extract_content,
    _resolve_deprecated_token,
    _safe_title,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_apify._types import CrawlerType


class ApifyDatasetLoader(BaseLoader, BaseModel):
    """Load datasets from Apify web scraping, crawling, and data extraction platform.

    To use, you should have the environment variable ``APIFY_TOKEN`` set
    with your API key, or pass ``apify_token`` as a named parameter to the
    constructor. ``APIFY_API_TOKEN`` is still accepted for backwards
    compatibility.

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

    apify_token: SecretStr | None = Field(
        default_factory=_apify_token_secret_factory,
        description='Apify API token. Falls back to the APIFY_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    _apify_client: ApifyClient = PrivateAttr()
    dataset_id: str
    """The ID of the dataset on the Apify platform."""
    dataset_mapping_function: Callable[[dict], Document]
    """A custom function that takes a single dictionary (an Apify dataset item)
     and converts it to an instance of the Document class."""

    def __init__(
        self,
        dataset_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        apify_token: str | SecretStr | None = None,
        *,
        apify_api_token: str | SecretStr | None = None,
    ) -> None:
        """Initialize the loader with an Apify dataset ID and a mapping function.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an instance
                of the Document class.
            apify_token (str | SecretStr): Apify API token. Falls back to the
                ``APIFY_TOKEN`` environment variable when *None*.
            apify_api_token: Deprecated alias for ``apify_token``.
        """
        apify_token = _resolve_deprecated_token(apify_token, apify_api_token)

        init_kwargs: dict[str, Any] = {
            'dataset_id': dataset_id,
            'dataset_mapping_function': dataset_mapping_function,
        }
        if apify_token is not None:
            init_kwargs['apify_token'] = apify_token
        super().__init__(**init_kwargs)

    @model_validator(mode='after')
    def _init_client(self) -> ApifyDatasetLoader:
        """Validate the resolved Apify token and initialise the client.

        The token default factory resolves ``APIFY_TOKEN`` first and
        ``APIFY_API_TOKEN`` as a legacy fallback.

        Returns:
            ApifyDatasetLoader: The validated loader instance.

        Raises:
            ValueError: If no token is available from any source.
        """
        if self.apify_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._apify_client = _create_apify_client(ApifyClient, self.apify_token.get_secret_value())
        return self

    def load(self) -> list[Document]:
        """Load documents.

        Returns:
            list[Document]: A list of mapped Document objects.
        """
        dataset_items = self._apify_client.dataset(self.dataset_id).list_items(clean=True).items
        return list(map(self.dataset_mapping_function, dataset_items))

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents.

        Yields:
            Document: A mapped Document object.
        """
        dataset_items = self._apify_client.dataset(self.dataset_id).iterate_items(
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
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.
        apify_api_token: Deprecated alias for ``apify_token``.
        max_crawl_pages: Maximum number of pages to crawl.
        max_crawl_depth: Maximum link-follow depth from the seed URL.
        crawler_type: Crawler engine (e.g. ``"cheerio"``, ``"playwright:firefox"``).
        timeout_secs: Maximum time in seconds to wait for the crawl.

    Returns:
        Iterator (or list) of ``Document`` objects.  ``page_content`` contains
        the page markdown; ``metadata`` includes ``source``, ``title``, and
        ``crawl_depth``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

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
        apify_token: str | SecretStr | None = None,
        *,
        apify_api_token: str | SecretStr | None = None,
        max_crawl_pages: int = _DEFAULT_MAX_CRAWL_PAGES,
        max_crawl_depth: int = _DEFAULT_MAX_CRAWL_DEPTH,
        crawler_type: CrawlerType = _DEFAULT_CRAWLER_TYPE,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
    ) -> None:
        apify_token = _resolve_deprecated_token(apify_token, apify_api_token)

        self.url = url
        self.max_crawl_pages = max_crawl_pages
        self.max_crawl_depth = max_crawl_depth
        self.crawler_type: CrawlerType = crawler_type
        self.timeout_secs = timeout_secs
        self._client = ApifyToolsClient(apify_token=apify_token)

    def lazy_load(self) -> Iterator[Document]:
        """Crawl the website and yield Documents.

        Yields:
            Document: One document per crawled page.
        """
        _, items = self._client.crawl_website(
            self.url,
            max_crawl_pages=self.max_crawl_pages,
            max_crawl_depth=self.max_crawl_depth,
            crawler_type=self.crawler_type,
            timeout_secs=self.timeout_secs,
        )
        for item in items:
            # Some Actor responses surface list-typed entries (e.g. nested
            # arrays for sitemap-style outputs). Skip anything non-dict.
            if not isinstance(item, dict):
                continue
            page_content = _extract_content(item)
            # website-content-crawler nests depth under crawl.depth; there is no
            # top-level crawlDepth field.
            crawl_meta = item.get('crawl')
            metadata: dict[str, Any] = {
                'source': item.get('url', ''),
                'title': _safe_title(item),
                'crawl_depth': crawl_meta.get('depth', 0) if isinstance(crawl_meta, dict) else 0,
            }
            yield Document(page_content=page_content, metadata=metadata)
