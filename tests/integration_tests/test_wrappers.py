from langchain_core.documents import Document

from langchain_apify import ApifyWrapper


def test_apify_wrapper_call_actor() -> None:
    apify = ApifyWrapper()

    loader = apify.call_actor(
        actor_id="apify/python-example",
        run_input={"first_number": 2, "second_number": 3},
        dataset_mapping_function=lambda item: Document(
            page_content=str(item["sum"]) or "",
        ),
    )
    documents = loader.load()

    assert documents[0].page_content == "5"
