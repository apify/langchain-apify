"""LangChain retrievers backed by Apify Actors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient, ApifyClientAsync
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import _create_apify_client

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

_RAG_WEB_BROWSER_ACTOR_ID = 'apify/rag-web-browser'
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

    max_results: int = Field(default=5, description='Maximum number of documents to return.')
    timeout_secs: int = Field(default=_DEFAULT_TIMEOUT_SECS, description='Maximum Actor run time in seconds.')

    _sync_client: ApifyClient = PrivateAttr()
    _async_client: ApifyClientAsync = PrivateAttr()

    def __init__(self, apify_api_token: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        token = apify_api_token or os.getenv('APIFY_API_TOKEN')
        if not token:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._sync_client = _create_apify_client(ApifyClient, token)
        self._async_client = _create_apify_client(ApifyClientAsync, token)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        run_input = {
            'query': query,
            'maxResults': self.max_results,
        }
        run = self._sync_client.actor(_RAG_WEB_BROWSER_ACTOR_ID).call(
            run_input=run_input,
            timeout_secs=self.timeout_secs,
            logger=None,
        )
        if run is None:
            return []

        dataset_id = run.get('defaultDatasetId')
        if not dataset_id:
            return []

        items = self._sync_client.dataset(dataset_id).list_items(
            limit=self.max_results, clean=True,
        ).items
        return self._items_to_documents(items)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        run_input = {
            'query': query,
            'maxResults': self.max_results,
        }
        run = await self._async_client.actor(_RAG_WEB_BROWSER_ACTOR_ID).call(
            run_input=run_input,
            timeout_secs=self.timeout_secs,
            logger=None,
        )
        if run is None:
            return []

        dataset_id = run.get('defaultDatasetId')
        if not dataset_id:
            return []

        items = (
            await self._async_client.dataset(dataset_id).list_items(
                limit=self.max_results, clean=True,
            )
        ).items
        return self._items_to_documents(items)

    @staticmethod
    def _items_to_documents(items: list[dict]) -> list[Document]:
        """Convert Actor dataset items to LangChain Documents."""
        docs: list[Document] = []
        for item in items:
            page_content = item.get('text') or item.get('markdown') or ''
            metadata: dict[str, Any] = {
                'source': item.get('crawledUrl') or item.get('url', ''),
                'title': item.get('metadata', {}).get('title', '') if isinstance(item.get('metadata'), dict) else '',
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
