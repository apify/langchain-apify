from unittest.mock import patch

from apify_client.clients import DatasetClient
from apify_shared.models import ListPage
from langchain_core.documents import Document

from langchain_apify import ApifyDatasetLoader


def test_apify_dataset_loader_load() -> None:
    with patch.object(DatasetClient, "list_items") as mock_list_items:
        mock_list_items.return_value = ListPage(
            data={"items": [{"text": "Apify is great!", "url": "https://apify.com"}]}
        )

        loader = ApifyDatasetLoader(
            apify_api_token="dummy-api-token",
            dataset_id="dummy-dataset-id",
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"], metadata={"source": item["url"]}
            ),
        )
        documents = loader.load()

        mock_list_items.assert_called_once()
        assert documents[0].page_content == "Apify is great!"
        assert documents[0].metadata["source"] == "https://apify.com"


def test_apify_dataset_loader_lazy_load() -> None:
    with patch.object(DatasetClient, "iterate_items") as mock_list_items:
        mock_list_items.return_value = iter(
            [{"text": "Apify is great!", "url": "https://apify.com"}]
        )

        loader = ApifyDatasetLoader(
            apify_api_token="dummy-api-token",
            dataset_id="dummy-dataset-id",
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"], metadata={"source": item["url"]}
            ),
        )
        documents = list(loader.lazy_load())

        mock_list_items.assert_called_once()
        assert documents[0].page_content == "Apify is great!"
        assert documents[0].metadata["source"] == "https://apify.com"
