from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_apify._client import ApifyToolsClient
from langchain_apify.retrievers import ApifySearchRetriever

RAG_ITEMS: list[dict] = [
    {
        'crawledUrl': 'https://example.com/1',
        'text': 'Page 1 content',
        'metadata': {'title': 'Page 1'},
    },
    {
        'crawledUrl': 'https://example.com/2',
        'text': 'Page 2 content',
        'metadata': {'title': 'Page 2'},
    },
]


def _make_retriever(mock_client: MagicMock, **kwargs: Any) -> ApifySearchRetriever:  # noqa: ANN401
    """Instantiate a retriever with a mocked ApifyToolsClient."""
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        retriever = ApifySearchRetriever(apify_api_token=SecretStr('dummy-token'), **kwargs)
    retriever._client = mock_client
    return retriever


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifySearchRetriever()


def test_init_with_explicit_token() -> None:
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        retriever = ApifySearchRetriever(apify_api_token=SecretStr('my-token'))
        assert retriever.max_results == 5
        assert retriever.timeout_secs == 300


def test_init_custom_params() -> None:
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        retriever = ApifySearchRetriever(apify_api_token=SecretStr('t'), max_results=3, timeout_secs=60)
        assert retriever.max_results == 3
        assert retriever.timeout_secs == 60


# ---------------------------------------------------------------------------
# Sync retrieval
# ---------------------------------------------------------------------------


def test_sync_returns_documents() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = RAG_ITEMS
    retriever = _make_retriever(mock_client, max_results=5)

    docs = retriever._get_relevant_documents('test query')

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == 'Page 1 content'
    assert docs[0].metadata['source'] == 'https://example.com/1'
    assert docs[0].metadata['title'] == 'Page 1'
    assert docs[1].page_content == 'Page 2 content'
    assert docs[1].metadata['source'] == 'https://example.com/2'


def test_sync_calls_helper_with_correct_args() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = []
    retriever = _make_retriever(mock_client, max_results=3, timeout_secs=60)

    retriever._get_relevant_documents('my search')

    mock_client.rag_web_search.assert_called_once_with(
        'my search',
        max_results=3,
        timeout_secs=60,
    )


def test_sync_empty_results() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = []
    retriever = _make_retriever(mock_client)

    docs = retriever._get_relevant_documents('test')

    assert docs == []


def test_sync_helper_failure_propagates() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.side_effect = RuntimeError(
        'Actor run run-bad ended with status FAILED.',
    )
    retriever = _make_retriever(mock_client)

    with pytest.raises(RuntimeError, match='FAILED'):
        retriever._get_relevant_documents('test')


# ---------------------------------------------------------------------------
# Async retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_returns_documents() -> None:
    """Async path wraps the sync helper via asyncio.to_thread."""
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = RAG_ITEMS
    retriever = _make_retriever(mock_client, max_results=5)

    docs = await retriever._aget_relevant_documents('test query')

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == 'Page 1 content'
    assert docs[0].metadata['source'] == 'https://example.com/1'


@pytest.mark.asyncio
async def test_async_calls_helper_with_correct_args() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = []
    retriever = _make_retriever(mock_client, max_results=3, timeout_secs=60)

    await retriever._aget_relevant_documents('my search')

    mock_client.rag_web_search.assert_called_once_with(
        'my search',
        max_results=3,
        timeout_secs=60,
    )


@pytest.mark.asyncio
async def test_async_empty_results() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.return_value = []
    retriever = _make_retriever(mock_client)

    docs = await retriever._aget_relevant_documents('test')

    assert docs == []


@pytest.mark.asyncio
async def test_async_helper_failure_propagates() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.rag_web_search.side_effect = RuntimeError(
        'Actor run run-bad ended with status FAILED.',
    )
    retriever = _make_retriever(mock_client)

    with pytest.raises(RuntimeError, match='FAILED'):
        await retriever._aget_relevant_documents('test')


# ---------------------------------------------------------------------------
# _items_to_documents edge cases
# ---------------------------------------------------------------------------


def test_items_to_documents_uses_url_fallback() -> None:
    items = [{'url': 'https://fallback.com', 'text': 'content', 'metadata': {'title': 'T'}}]

    docs = ApifySearchRetriever._items_to_documents(items)

    assert docs[0].metadata['source'] == 'https://fallback.com'


def test_items_to_documents_uses_markdown_fallback() -> None:
    items = [{'crawledUrl': 'https://example.com', 'markdown': '# MD content', 'metadata': {'title': 'T'}}]

    docs = ApifySearchRetriever._items_to_documents(items)

    assert docs[0].page_content == '# MD content'


def test_items_to_documents_missing_metadata() -> None:
    items = [{'crawledUrl': 'https://example.com', 'text': 'content'}]

    docs = ApifySearchRetriever._items_to_documents(items)

    assert docs[0].metadata['title'] == ''
    assert docs[0].metadata['source'] == 'https://example.com'


def test_items_to_documents_non_dict_metadata() -> None:
    items = [{'crawledUrl': 'https://example.com', 'text': 'content', 'metadata': 'not-a-dict'}]

    docs = ApifySearchRetriever._items_to_documents(items)

    assert docs[0].metadata['title'] == ''
