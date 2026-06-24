"""LangChain retrievers backed by Apify Actors."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr, SecretStr, model_validator

from langchain_apify._client import ApifyToolsClient
from langchain_apify._constants import _DEFAULT_RAG_MAX_RESULTS, _DEFAULT_RUN_TIMEOUT_SECS
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _apify_token_secret_factory,
    _extract_content,
    _extract_source,
    _resolve_deprecated_token_values,
    _safe_title,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )


class ApifySearchRetriever(BaseRetriever):
    """Retrieve documents from the web for RAG using Apify.

    Wraps the ``apify/rag-web-browser`` Actor.  Each invocation runs a web
    search, crawls the top results, and returns their content as LangChain
    ``Document`` objects ready for a RAG pipeline.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.
        apify_api_token: Deprecated alias for ``apify_token``.
        max_results: Maximum number of ``Document`` objects to return per query.
        timeout_secs: Maximum time in seconds to wait for the Actor run.

    Returns:
        List of ``Document`` objects.  ``page_content`` contains the crawled
        text; ``metadata`` includes ``source`` (URL) and ``title``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifySearchRetriever

            retriever = ApifySearchRetriever(max_results=3)
            docs = retriever.invoke("What is LangChain?")
    """

    apify_token: SecretStr | None = Field(
        default_factory=_apify_token_secret_factory,
        description='Apify API token. Falls back to the APIFY_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    max_results: int = Field(default=_DEFAULT_RAG_MAX_RESULTS, description='Maximum number of documents to return.')
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description='Maximum Actor run time in seconds.')

    _client: ApifyToolsClient = PrivateAttr()

    @model_validator(mode='before')
    @classmethod
    def _handle_deprecated_apify_api_token(cls, values: dict) -> dict:
        return _resolve_deprecated_token_values(values)

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Construct the underlying ``ApifyToolsClient``.

        Mirrors ``_ApifyGenericTool``: guard against a missing token locally
        before constructing the client, so the failure mode is consistent
        across tools and the retriever.
        """
        if self.apify_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._client = ApifyToolsClient(apify_token=self.apify_token.get_secret_value())
        super().model_post_init(context)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        _, items = self._client.rag_web_search(
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
            self._client.rag_web_search,
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
            page_content = _extract_content(item)
            metadata: dict[str, Any] = {
                'source': _extract_source(item),
                'title': _safe_title(item),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
