from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from langchain_apify.tools import ApifyActorsTool
from langchain_apify.utils import actor_id_to_tool_name

if TYPE_CHECKING:
    from collections.abc import Generator


def test_apify_actors_tool_instance() -> None:
    """Tests the ApifyActorsTool instance creation.

    Creates an instance of the ApifyActorsTool and
        checks if the instance is created correctly.
    """
    with (
        patch.object(
            ApifyActorsTool,
            '_create_description',
            return_value='Mocked description',
        ),
        patch.object(
            ApifyActorsTool,
            '_build_tool_args_schema_model',
        ) as mock_build_tool_args_schema_model,
    ):

        class DummyModel(BaseModel):
            run_input: str

        mock_build_tool_args_schema_model.return_value = DummyModel

        actor_id = 'apify/python-example'
        tool = ApifyActorsTool(actor_id=actor_id, apify_api_token='dummy-token')
        assert isinstance(tool, ApifyActorsTool)
        assert tool.description == 'Mocked description'
        assert tool.name == actor_id_to_tool_name(actor_id)
        assert tool.args_schema == DummyModel


def test_run_actor_method(apify_actors_tool_fixture: ApifyActorsTool) -> None:
    """Tests the ApifyActorsTool._run_actor method.

    Mocks the ApifyActorsTool._run_actor method to return a single item.
    """
    with patch.object(ApifyActorsTool, '_run_actor') as mock_run_actor:
        mock_run_actor.return_value = [{'text': 'Apify is great!'}]

        result = apify_actors_tool_fixture.invoke(
            input={'run_input': {'query': 'what is Apify?', 'maxResults': 3}},
        )
        mock_run_actor.assert_called_once()
        assert result[0]['text'] == 'Apify is great!'


@pytest.fixture
def apify_actors_tool_fixture() -> Generator[ApifyActorsTool, None, None]:
    """Fixture to create an instance of the ApifyActorsTool.

    Yields:
        ApifyActorsTool: An instance of the ApifyActorsTool.
    """
    with (
        patch.object(
            ApifyActorsTool,
            '_create_description',
            return_value='Mocked description',
        ),
        patch.object(
            ApifyActorsTool,
            '_build_tool_args_schema_model',
        ) as mock_build_tool_args_schema_model,
    ):

        class DummyModel(BaseModel):
            run_input: str | dict

        mock_build_tool_args_schema_model.return_value = DummyModel

        tool = ApifyActorsTool(actor_id='apify/python-example', apify_api_token='dummy-token')
        yield tool
