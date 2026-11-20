"""LangChain retrievers backed by Apify Actors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr, SecretStr, model_validator

from langchain_apify._client import ApifyToolsClient
from langchain_apify._constants import (
    _DEFAULT_RAG_MAX_RESULTS,
    _DEFAULT_RUN_TIMEOUT_SECS,
    _RAG_MAX_RESULTS_CAP,
)
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _apify_token_secret_factory,
    _extract_content,
    _extract_source,
    _resolve_deprecated_token_values,
    _safe_title,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        max_total_charge_usd: Maximum total cost in USD of each Actor run, or
            *None* for no limit. A limit below the Actor's own minimum makes
            the run fail.

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
    # Frozen, because the client holds the limit; `model_copy` builds a new client for a copy with another limit.
    max_total_charge_usd: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        frozen=True,
        description='Maximum total cost in USD of each Actor run, or None for no limit.',
    )

    _client: ApifyToolsClient = PrivateAttr()

    @model_validator(mode='before')
    @classmethod
    def _handle_deprecated_apify_api_token(cls, values: dict) -> dict:
        return _resolve_deprecated_token_values(values)

    def model_post_init(self, context: Any) -> None:
        """Construct the underlying ``ApifyToolsClient``.

        Mirrors ``_ApifyGenericTool``: guard against a missing token locally
        before constructing the client, so the failure mode is consistent
        across tools and the retriever.
        """
        self._client = self._new_client()
        super().model_post_init(context)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy the retriever; the copy gets its own client, with its own token and charge limit."""
        copy = super().model_copy(update=update, deep=deep)
        # The client holds the token and limit it was created with, so the copy needs its own.
        copy._client = copy._new_client()  # noqa: SLF001 - the copy is an instance of this class
        return copy

    def _new_client(self) -> ApifyToolsClient:
        if self.apify_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        return ApifyToolsClient(
            apify_token=self.apify_token.get_secret_value(), max_total_charge_usd=self.max_total_charge_usd
        )

    def _clamped_max_results(self) -> int:
        """Clamp ``max_results`` to the rag-web-browser Actor's ceiling.

        The Actor rejects ``maxResults`` above :data:`_RAG_MAX_RESULTS_CAP` at
        runtime, so clamp here rather than let the request fail outright.
        """
        return max(1, min(self.max_results, _RAG_MAX_RESULTS_CAP))

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        _, items = self._client.rag_web_search(
            query,
            max_results=self._clamped_max_results(),
            timeout_secs=self.timeout_secs,
        )
        return self._items_to_documents(items)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        _, items = await self._client.arag_web_search(
            query,
            max_results=self._clamped_max_results(),
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
