from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from apify_client._types import ListPage
from apify_client.clients import DatasetClient
from langchain_core.documents import Document

from langchain_apify import ApifyCrawlLoader, ApifyDatasetLoader
from langchain_apify._client import ApifyToolsClient


def test_apify_dataset_loader_load() -> None:
    """Tests the ApifyDatasetLoader.load method.

    Mocks the ApifyClient and DatasetClient to return a single item.
    """
    with patch.object(DatasetClient, 'list_items') as mock_list_items:
        mock_list_items.return_value = ListPage(
            data={'items': [{'text': 'Apify is great!', 'url': 'https://apify.com'}]},
        )

        loader = ApifyDatasetLoader(
            apify_api_token='dummy-token',
            dataset_id='dummy-dataset-id',
            dataset_mapping_function=lambda item: Document(
                page_content=item['text'],
                metadata={'source': item['url']},
            ),
        )
        documents = loader.load()

        mock_list_items.assert_called_once()
        assert documents[0].page_content == 'Apify is great!'
        assert documents[0].metadata['source'] == 'https://apify.com'


def test_apify_dataset_loader_lazy_load() -> None:
    """Tests the ApifyDatasetLoader.lazy_load method.

    Mocks the ApifyClient and DatasetClient to return a single item.
    """
    with patch.object(DatasetClient, 'iterate_items') as mock_list_items:
        mock_list_items.return_value = iter(
            [{'text': 'Apify is great!', 'url': 'https://apify.com'}],
        )

        loader = ApifyDatasetLoader(
            apify_api_token='dummy-token',
            dataset_id='dummy-dataset-id',
            dataset_mapping_function=lambda item: Document(
                page_content=item['text'],
                metadata={'source': item['url']},
            ),
        )
        documents = list(loader.lazy_load())

        mock_list_items.assert_called_once()
        assert documents[0].page_content == 'Apify is great!'
        assert documents[0].metadata['source'] == 'https://apify.com'


# ---------------------------------------------------------------------------
# ApifyCrawlLoader
# ---------------------------------------------------------------------------

CRAWL_ITEMS: list[dict] = [
    {
        'url': 'https://example.com/',
        'markdown': '# Home',
        'text': 'Home',
        'metadata': {'title': 'Home Page'},
        'crawlDepth': 0,
    },
    {
        'url': 'https://example.com/about',
        'markdown': '# About',
        'text': 'About',
        'metadata': {'title': 'About Page'},
        'crawlDepth': 1,
    },
]


def _make_crawl_loader(
    mock_client: MagicMock,
    **kwargs: object,
) -> ApifyCrawlLoader:
    with patch.object(ApifyToolsClient, '__init__', return_value=None):
        loader = ApifyCrawlLoader(url='https://example.com', apify_api_token='dummy', **kwargs)
    loader._client = mock_client
    return loader


def test_crawl_loader_lazy_load() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = CRAWL_ITEMS
    loader = _make_crawl_loader(mock_client)

    docs = list(loader.lazy_load())

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == '# Home'
    assert docs[0].metadata['source'] == 'https://example.com/'
    assert docs[0].metadata['title'] == 'Home Page'
    assert docs[0].metadata['crawl_depth'] == 0
    assert docs[1].page_content == '# About'
    assert docs[1].metadata['crawl_depth'] == 1


def test_crawl_loader_load_delegates_to_lazy_load() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = CRAWL_ITEMS
    loader = _make_crawl_loader(mock_client)

    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == '# Home'


def test_crawl_loader_passes_params() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = []
    loader = _make_crawl_loader(
        mock_client,
        max_crawl_pages=5,
        max_crawl_depth=2,
        crawler_type='playwright',
        timeout_secs=120,
    )

    list(loader.lazy_load())

    mock_client.crawl_website.assert_called_once_with(
        'https://example.com',
        max_crawl_pages=5,
        max_crawl_depth=2,
        crawler_type='playwright',
        timeout_secs=120,
    )


def test_crawl_loader_empty_results() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = []
    loader = _make_crawl_loader(mock_client)

    docs = loader.load()

    assert docs == []


def test_crawl_loader_text_fallback() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = [
        {'url': 'https://example.com/', 'text': 'Plain text', 'metadata': {'title': 'T'}},
    ]
    loader = _make_crawl_loader(mock_client)

    docs = list(loader.lazy_load())

    assert docs[0].page_content == 'Plain text'


def test_crawl_loader_missing_metadata() -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.return_value = [
        {'url': 'https://example.com/', 'markdown': '# Content'},
    ]
    loader = _make_crawl_loader(mock_client)

    docs = list(loader.lazy_load())

    assert docs[0].metadata['title'] == ''
    assert docs[0].metadata['crawl_depth'] == 0


def test_crawl_loader_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyCrawlLoader(url='https://example.com')


def test_crawl_loader_failure_raises(mock_tools_client: MagicMock) -> None:
    mock_client = MagicMock(spec=ApifyToolsClient)
    mock_client.crawl_website.side_effect = RuntimeError('Actor run run-bad ended with status FAILED.')
    loader = _make_crawl_loader(mock_client)

    with pytest.raises(RuntimeError, match='FAILED'):
        loader.load()


def test_apify_dataset_loader_apify_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loader should accept APIFY_TOKEN as a secondary env-var fallback."""
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    monkeypatch.setenv('APIFY_TOKEN', 'platform-token')

    with patch.object(DatasetClient, 'list_items') as mock_list_items:
        mock_list_items.return_value = ListPage(data={'items': []})
        loader = ApifyDatasetLoader(
            dataset_id='d',
            dataset_mapping_function=lambda _item: Document(page_content='x'),
        )
        assert loader.load() == []


def test_apify_dataset_loader_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    monkeypatch.delenv('APIFY_TOKEN', raising=False)
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyDatasetLoader(
            dataset_id='d',
            dataset_mapping_function=lambda _item: Document(page_content='x'),
        )
