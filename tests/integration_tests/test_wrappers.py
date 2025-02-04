from langchain_core.documents import Document

from langchain_apify import ApifyWrapper


def test_apify_wrapper_call_actor() -> None:
    """Test ApifyWrapper.call_actor method by calling the python-example actor.

    This actor takes two numbers as input and returns their sum, so it is deterministic
    and we can easily verify that the method works as expected with the Apify platform.
    """
    apify = ApifyWrapper()

    loader = apify.call_actor(
        actor_id='apify/python-example',
        run_input={'first_number': 2, 'second_number': 3},
        dataset_mapping_function=lambda item: Document(
            page_content=str(item['sum']) or '',
        ),
    )
    documents = loader.load()

    assert documents[0].page_content == '5'
