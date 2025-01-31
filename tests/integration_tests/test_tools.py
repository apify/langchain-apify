import json
from typing import Type

from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_apify.tools import ApifyActorsTool


class TestApifyActorsToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ApifyActorsTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"actor_id": "apify/python-example"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"run_input": json.dumps({"first_number": 2, "second_number": 3})}
