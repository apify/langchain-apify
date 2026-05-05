"""LangChain retrievers backed by Apify Actors."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import secret_from_env
from pydantic import Field, PrivateAttr, SecretStr

from langchain_apify._client import ApifyToolsClient

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

_DEFAULT_TIMEOUT_SECS = 300


class ApifySearchRetriever(BaseRetriever):
    """Retrieve documents from the web for RAG using Apify.

    Wraps the ``apify/rag-web-browser`` Actor.  Each invocation runs a web
    search, crawls the top results, and returns their content as LangChain
    ``Document`` objects ready for a RAG pipeline.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.
        max_results: Maximum number of ``Document`` objects to return per query.
        timeout_secs: Maximum time in seconds to wait for the Actor run.

    Returns:
        List of ``Document`` objects.  ``page_content`` contains the crawled
        text; ``metadata`` includes ``source`` (URL) and ``title``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifySearchRetriever

            retriever = ApifySearchRetriever(max_results=3)
            docs = retriever.invoke("What is LangChain?")
    """

    apify_api_token: SecretStr | None = Field(
        default_factory=secret_from_env('APIFY_API_TOKEN', default=None),
        description='Apify API token. Falls back to the APIFY_API_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    max_results: int = Field(default=5, description='Maximum number of documents to return.')
    timeout_secs: int = Field(default=_DEFAULT_TIMEOUT_SECS, description='Maximum Actor run time in seconds.')

    _client: ApifyToolsClient = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Construct the underlying ``ApifyToolsClient``.

        The helper handles ``None`` / ``SecretStr`` / env-fallback and raises
        ``ValueError`` if no token is available.
        """
        self._client = ApifyToolsClient(apify_api_token=self.apify_api_token)
        super().model_post_init(context)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        _, items = self._client.rag_web_browser_search(
            query,
            max_results=self.max_results,
            timeout_secs=self.timeout_secs,
        )
        return self._items_to_documents(items)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        # ApifyToolsClient is sync-only.
        _, items = await asyncio.to_thread(
            self._client.rag_web_browser_search,
            query,
            max_results=self.max_results,
            timeout_secs=self.timeout_secs,
        )
        return self._items_to_documents(items)

    @staticmethod
    def _items_to_documents(items: list[dict]) -> list[Document]:
        """Convert Actor dataset items to LangChain Documents."""
        docs: list[Document] = []
        for item in items:
            page_content = item.get('text') or item.get('markdown') or ''
            raw_meta = item.get('metadata')
            item_metadata: dict = raw_meta if isinstance(raw_meta, dict) else {}
            metadata: dict[str, Any] = {
                # apify/rag-web-browser nests url/title under "metadata"; older
                # Actors and tests use top-level keys. Both are supported.
                'source': item.get('crawledUrl') or item.get('url') or item_metadata.get('url', ''),
                'title': item_metadata.get('title', ''),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
