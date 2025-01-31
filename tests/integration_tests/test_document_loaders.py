import os
from typing import Iterator

from apify_client import ApifyClient
from langchain_core.documents import Document

from langchain_apify import ApifyDatasetLoader


def test_apify_dataset_loader_load() -> None:
    token = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(token=token)

    dataset_name = "langchain-test-apify-dataset-loader-load"

    existing_datasets = client.datasets().list().items
    for dataset in existing_datasets:
        if dataset["name"] == dataset_name:
            client.dataset(dataset["id"]).delete()

    dataset = client.datasets().get_or_create(name=dataset_name)

    dataset_client = client.dataset(dataset["id"])

    dataset_client.push_items([{"text": "Hello, world!", "url": "https://example.com"}])
    dataset_client.push_items(
        [{"text": "Goodbye, world!", "url": "https://example.com"}]
    )

    loader = ApifyDatasetLoader(
        dataset_id=dataset["id"],
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
        ),
    )
    documents = loader.load()

    assert documents[0].page_content == "Hello, world!"
    assert documents[0].metadata["source"] == "https://example.com"
    assert documents[1].page_content == "Goodbye, world!"
    assert documents[1].metadata["source"] == "https://example.com"


def test_apify_dataset_loader_lazy_load() -> None:
    token = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(token=token)

    dataset_name = "langchain-test-apify-dataset-loader-lazy-load"

    existing_datasets = client.datasets().list().items
    for dataset in existing_datasets:
        if dataset["name"] == dataset_name:
            client.dataset(dataset["id"]).delete()

    dataset = client.datasets().get_or_create(name=dataset_name)

    dataset_client = client.dataset(dataset["id"])

    dataset_client.push_items(
        [{"text": "Hello, lazy world!", "url": "https://example.com"}]
    )
    dataset_client.push_items(
        [{"text": "Goodbye, lazy world!", "url": "https://example.com"}]
    )

    loader = ApifyDatasetLoader(
        dataset_id=dataset["id"],
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
        ),
    )
    documents = loader.lazy_load()
    assert isinstance(documents, Iterator)

    documents_list = list(documents)
    assert documents_list[0].page_content == "Hello, lazy world!"
    assert documents_list[0].metadata["source"] == "https://example.com"
    assert documents_list[1].page_content == "Goodbye, lazy world!"
    assert documents_list[1].metadata["source"] == "https://example.com"
