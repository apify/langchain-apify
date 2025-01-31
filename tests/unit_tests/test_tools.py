from typing import Generator
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from langchain_apify.tools import ApifyActorsTool
from langchain_apify.utils import actor_id_to_tool_name


def test_apify_actors_tool_instance() -> None:
    with patch.object(
        ApifyActorsTool, "create_description", return_value="Mocked description"
    ), patch.object(
        ApifyActorsTool, "build_tool_args_schema_model"
    ) as mock_build_tool_args_schema_model:

        class DummyModel(BaseModel):
            run_input: str

        mock_build_tool_args_schema_model.return_value = DummyModel

        actor_id = "apify/python-example"
        tool = ApifyActorsTool(actor_id=actor_id, apify_api_token="token")
        assert isinstance(tool, ApifyActorsTool)
        assert tool.description == "Mocked description"
        assert tool.name == actor_id_to_tool_name(actor_id)
        assert tool.args_schema == DummyModel


def test_run_actor_method(apify_actors_tool_fixture: ApifyActorsTool) -> None:
    with patch.object(ApifyActorsTool, "_run_actor") as mock_run_actor:
        mock_run_actor.return_value = [{"text": "monero is based"}]

        result = apify_actors_tool_fixture.invoke(
            input={"run_input": {"query": "what is monero?", "maxResults": 3}}
        )
        mock_run_actor.assert_called_once()
        assert result[0]["text"] == "monero is based"


@pytest.fixture
def apify_actors_tool_fixture() -> Generator[ApifyActorsTool, None, None]:
    with patch.object(
        ApifyActorsTool, "create_description", return_value="Mocked description"
    ), patch.object(
        ApifyActorsTool, "build_tool_args_schema_model"
    ) as mock_build_tool_args_schema_model:

        class DummyModel(BaseModel):
            run_input: str | dict

        mock_build_tool_args_schema_model.return_value = DummyModel

        tool = ApifyActorsTool(actor_id="apify/python-example", apify_api_token="token")
        yield tool
