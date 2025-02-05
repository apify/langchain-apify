from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_apify.tools import ApifyActorsTool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class TestApifyActorsToolIntegration(ToolsIntegrationTests):
    """Integration tests for the ApifyActorsTool.

    This test class inherits from ToolsIntegrationTests, which provides a base
    class for tools integration tests. It includes the following attributes and methods:

    Tests run in this class:
        test_async_invoke_matches_output_schema(tool): Tests if the tool returns a valid
            ToolMessage content when invoked asynchronously with a ToolCall.
        test_async_invoke_no_tool_call(tool): Tests if the tool can return anything
            without throwing an error when invoked asynchronously without a ToolCall.
        test_invoke_matches_output_schema(tool): Tests if the tool returns a valid
            ToolMessage content when invoked with a ToolCall.
        test_invoke_no_tool_call(tool): Tests if the tool can return anything without
            throwing an error when invoked without a ToolCall.
    """

    @property
    def tool_constructor(self) -> type[BaseTool]:
        """Return the tool class to be tested."""
        return ApifyActorsTool

    @property
    def tool_constructor_params(self) -> dict:
        """Return the parameters to pass to the tool constructor."""
        return {'actor_id': 'apify/python-example'}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Return an example of the parameters to pass to the tool invoke method."""
        return {'run_input': json.dumps({'first_number': 2, 'second_number': 3})}
