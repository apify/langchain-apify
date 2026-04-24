from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

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


def _make_retriever(
    mock_sync_client: MagicMock,
    mock_async_client: MagicMock | None = None,
    **kwargs: object,
) -> ApifySearchRetriever:
    """Create a retriever with mocked Apify clients."""
    with (
        patch('langchain_apify.retrievers._create_apify_client') as mock_create,
    ):
        mock_create.side_effect = [mock_sync_client, mock_async_client or MagicMock()]
        return ApifySearchRetriever(apify_api_token='dummy-token', **kwargs)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifySearchRetriever()


def test_init_with_explicit_token() -> None:
    with patch('langchain_apify.retrievers._create_apify_client'):
        retriever = ApifySearchRetriever(apify_api_token='my-token')
        assert retriever.max_results == 5
        assert retriever.timeout_secs == 300


def test_init_custom_params() -> None:
    with patch('langchain_apify.retrievers._create_apify_client'):
        retriever = ApifySearchRetriever(apify_api_token='t', max_results=3, timeout_secs=60)
        assert retriever.max_results == 3
        assert retriever.timeout_secs == 60


# ---------------------------------------------------------------------------
# _get_relevant_documents (sync)
# ---------------------------------------------------------------------------


def test_sync_returns_documents() -> None:
    mock_client = MagicMock()
    mock_client.actor.return_value.call.return_value = {
        'id': 'run-1',
        'status': 'SUCCEEDED',
        'defaultDatasetId': 'ds-1',
    }
    mock_client.dataset.return_value.list_items.return_value.items = RAG_ITEMS
    retriever = _make_retriever(mock_client, max_results=5)

    docs = retriever._get_relevant_documents('test query')

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == 'Page 1 content'
    assert docs[0].metadata['source'] == 'https://example.com/1'
    assert docs[0].metadata['title'] == 'Page 1'
    assert docs[1].page_content == 'Page 2 content'
    assert docs[1].metadata['source'] == 'https://example.com/2'


def test_sync_passes_correct_input() -> None:
    mock_client = MagicMock()
    mock_client.actor.return_value.call.return_value = {
        'defaultDatasetId': 'ds-1',
    }
    mock_client.dataset.return_value.list_items.return_value.items = []
    retriever = _make_retriever(mock_client, max_results=3, timeout_secs=60)

    retriever._get_relevant_documents('my search')

    mock_client.actor.return_value.call.assert_called_once_with(
        run_input={'query': 'my search', 'maxResults': 3},
        timeout_secs=60,
    )
    mock_client.dataset.return_value.list_items.assert_called_once_with(
        limit=3, clean=True,
    )


def test_sync_empty_results() -> None:
    mock_client = MagicMock()
    mock_client.actor.return_value.call.return_value = {
        'defaultDatasetId': 'ds-1',
    }
    mock_client.dataset.return_value.list_items.return_value.items = []
    retriever = _make_retriever(mock_client)

    docs = retriever._get_relevant_documents('test')

    assert docs == []


def test_sync_none_run_returns_empty() -> None:
    mock_client = MagicMock()
    mock_client.actor.return_value.call.return_value = None
    retriever = _make_retriever(mock_client)

    docs = retriever._get_relevant_documents('test')

    assert docs == []


def test_sync_no_dataset_id_returns_empty() -> None:
    mock_client = MagicMock()
    mock_client.actor.return_value.call.return_value = {'id': 'run-1', 'defaultDatasetId': None}
    retriever = _make_retriever(mock_client)

    docs = retriever._get_relevant_documents('test')

    assert docs == []


# ---------------------------------------------------------------------------
# _aget_relevant_documents (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_returns_documents() -> None:
    mock_async = MagicMock()
    mock_async.actor.return_value.call = AsyncMock(return_value={
        'id': 'run-1',
        'status': 'SUCCEEDED',
        'defaultDatasetId': 'ds-1',
    })
    mock_list_items = AsyncMock()
    mock_list_items.return_value.items = RAG_ITEMS
    mock_async.dataset.return_value.list_items = mock_list_items

    retriever = _make_retriever(MagicMock(), mock_async, max_results=5)

    docs = await retriever._aget_relevant_documents('test query')

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == 'Page 1 content'
    assert docs[0].metadata['source'] == 'https://example.com/1'


@pytest.mark.asyncio
async def test_async_none_run_returns_empty() -> None:
    mock_async = MagicMock()
    mock_async.actor.return_value.call = AsyncMock(return_value=None)
    retriever = _make_retriever(MagicMock(), mock_async)

    docs = await retriever._aget_relevant_documents('test')

    assert docs == []


@pytest.mark.asyncio
async def test_async_no_dataset_id_returns_empty() -> None:
    mock_async = MagicMock()
    mock_async.actor.return_value.call = AsyncMock(return_value={'defaultDatasetId': None})
    retriever = _make_retriever(MagicMock(), mock_async)

    docs = await retriever._aget_relevant_documents('test')

    assert docs == []


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
