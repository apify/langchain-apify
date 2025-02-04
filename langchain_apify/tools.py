from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Union

from apify_client import ApifyClient
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from langchain_apify.utils import (
    actor_id_to_tool_name,
    create_apify_client,
    get_actor_latest_build,
    prune_actor_input_schema,
)

from .const import MAX_DESCRIPTION_LEN

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        CallbackManagerForToolRun,
    )


class ApifyActorsTool(BaseTool):  # type: ignore[override, override]
    """Tool that runs Apify Actors.

    To use, you should have the environment variable `APIFY_API_TOKEN` set
    with your API key, or pass `apify_api_token`
    as a named parameter to the constructor.

    For details, see https://docs.apify.com/platform/integrations/langchain

    Example:
        .. code-block:: python

            from langchain_apify import ApifyActorsTool
            from langgraph.prebuilt import create_react_agent

            tool = ApifyActorsTool(actor_id="apify/rag-web-browser")
            # Use the tool directly to call the Apify Actor
            result = tool.invoke(
                input={"run_input": {"query": "what is Apify?", "maxResults": 3}}
            )

            # Use the tool with an agent
            tools = [tool]
            agent = create_react_agent(model, tools)

            for chunk in agent.stream(
                {"messages": [("human", "search for what is Apify?")]},
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
    """

    def __init__(
        self,
        actor_id: str,
        apify_api_token: str | None = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the tool with an Apify Actor.

        Args:
            actor_id (str): Actor name from Apify store to run.
            apify_api_token (Optional[str]): Apify API token.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        apify_api_token = apify_api_token or os.getenv('APIFY_API_TOKEN')
        if not apify_api_token:
            msg = 'APIFY_API_TOKEN environment variable is not set.'
            raise ValueError(msg)

        apify_client = create_apify_client(ApifyClient, apify_api_token)

        kwargs.update(
            {
                'name': actor_id_to_tool_name(actor_id),
                'description': self._create_description(apify_client, actor_id),
                'args_schema': self._build_tool_args_schema_model(
                    apify_client,
                    actor_id,
                ),
            },
        )

        super().__init__(*args, **kwargs)

        self._apify_client = apify_client
        self._actor_id = actor_id

    def _run(
        self,
        run_input: str | dict,
        _: CallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        """Use the Apify Actor.

        Args:
            run_input (Union[str, dict]): JSON input for the Actor.
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager for the
                tool run.

        Returns:
            list[dict]: The output dataset.
        """
        input_dict = json.loads(run_input) if isinstance(run_input, str) else run_input
        # retrieve if nested, just in case
        input_dict = input_dict.get('run_input', input_dict)
        return self._run_actor(input_dict)

    def _create_description(self, apify_client: ApifyClient, actor_id: str) -> str:
        """Create a description for the tool.

        Args:
            apify_client (ApifyClient): Apify client instance.
            actor_id (str): Actor name from Apify store to run.
        """
        build = get_actor_latest_build(apify_client, actor_id)
        readme = build.get('readme', '')
        if len(readme) > MAX_DESCRIPTION_LEN:
            readme = readme[:MAX_DESCRIPTION_LEN] + '...(TRUNCATED, README TOO LONG)'
        return (
            'Run an Apify Actor with the given input. '
            'Here README for available Apify Actor:\n\n'
            f'{readme}\n\n'
        )

    def _build_tool_args_schema_model(
        self,
        apify_client: ApifyClient,
        actor_id: str,
    ) -> type[BaseModel]:
        """Build a tool class for agent that runs the Apify Actor.

        Args:
            apify_client (ApifyClient): Apify client instance.
            actor_id (str): Actor name from Apify store to run.
        """
        build = get_actor_latest_build(apify_client, actor_id)
        if not (actor_input := build.get('actorDefinition', {}).get('input')):
            msg = 'Input schema not found'
            raise ValueError(msg)

        properties, required = prune_actor_input_schema(actor_input)
        properties = {'run_input': properties}

        _description = [
            (
                "JSON encoded as string with input schema "
                "(STRICTLY FOLLOW JSON FORMAT AND SCHEMA):\n\n"
                f"{json.dumps(properties, separators=(',', ':'))}"
                "\nIF THE TOOL INPUT SCHEMA SUPPORTS IT LIMIT THE NUMBER OF RESULTS"
            ),
        ]
        if required:
            _description.append('\n\nRequired fields:\n' + '\n'.join(required))
        description = ''.join(_description)

        return create_model(
            'ApifyActorsToolInput',
            run_input=(Union[str, dict], Field(..., description=description)),
        )

    def _run_actor(self, run_input: dict) -> list[dict]:
        """Run an Apify Actor and return the output dataset.

        Args:
            actor_id: str, Actor name from Apify store to run.
            run_input: dict, JSON input for the Actor.
            fields: list, List of fields to extract from the dataset.
                Other fields will be ignored.
        """
        if (
            details := self._apify_client.actor(actor_id=self._actor_id).call(
                run_input=run_input,
            )
        ) is None:
            msg = 'Actor run details not found'
            raise ValueError(msg)
        if (run_id := details.get('id')) is None:
            msg = 'Run id not found'
            raise ValueError(msg)
        run = self._apify_client.run(run_id=run_id)

        return run.dataset().list_items(clean=True).items
