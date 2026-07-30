"""Legacy dynamic-actor tool.

:class:`ApifyActorsTool` builds its argument schema and description at
construction time from a single Apify Actor's build, then runs that Actor.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, create_model

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _MAX_DESCRIPTION_LEN,
    _actor_id_to_tool_name,
    _create_apify_client,
    _get_actor_latest_build,
    _prune_actor_input_schema,
    _resolve_apify_token,
    _resolve_deprecated_token,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        CallbackManagerForToolRun,
    )


class ApifyActorsTool(BaseTool):  # type: ignore[override, override]
    """Tool that runs Apify Actors.

    To use, you should have the environment variable ``APIFY_TOKEN`` set
    with your API key, or pass ``apify_token`` as a named parameter to the
    constructor.

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

    _apify_client: ApifyClient = PrivateAttr()
    _actor_id: str = PrivateAttr()

    def __init__(
        self,
        actor_id: str,
        apify_token: str | SecretStr | None = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the tool with an Apify Actor.

        Args:
            actor_id (str): Actor name from Apify store to run.
            apify_token (Optional[str]): Apify API token.
            apify_api_token: Deprecated alias for ``apify_token``.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the ``APIFY_TOKEN`` environment variable is not set
        """
        if 'apify_api_token' in kwargs:
            apify_token = _resolve_deprecated_token(apify_token, kwargs.pop('apify_api_token'))

        _raw_token: str | None = (
            apify_token.get_secret_value()
            if isinstance(apify_token, SecretStr)
            else apify_token or _resolve_apify_token()
        )
        if not _raw_token:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)

        apify_client = _create_apify_client(ApifyClient, _raw_token)
        build = _get_actor_latest_build(apify_client, actor_id)

        kwargs.update(
            {
                'name': _actor_id_to_tool_name(actor_id),
                'description': self._create_description(build),
                'args_schema': self._build_tool_args_schema_model(build, actor_id),
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

        Returns:
            list[dict]: The output dataset.
        """
        input_dict = json.loads(run_input) if isinstance(run_input, str) else run_input
        # retrieve if nested, just in case
        input_dict = input_dict.get('run_input', input_dict)
        return self._run_actor(input_dict)

    @staticmethod
    def _create_description(build: dict) -> str:
        """Create a description for the tool from an Actor build.

        Args:
            build (dict): The Actor build, as returned by ``_get_actor_latest_build``.

        Returns:
            str: The description.
        """
        actor_description = build.get('actorDefinition', {}).get('description', '')
        if len(actor_description) > _MAX_DESCRIPTION_LEN:
            actor_description = actor_description[:_MAX_DESCRIPTION_LEN] + '...(TRUNCATED, TOO LONG)'
        return actor_description

    @staticmethod
    def _build_tool_args_schema_model(build: dict, actor_id: str) -> type[BaseModel]:
        """Build a tool class for an agent that runs the Apify Actor.

        Args:
            build (dict): The Actor build, as returned by ``_get_actor_latest_build``.
            actor_id (str): Actor name from Apify store to run (used for error messages).

        Returns:
            type[BaseModel]: The tool input model class for the Apify Actor.

        Raises:
            ValueError: If the input schema is not found in the Actor build.
        """
        if not (actor_input := build.get('actorDefinition', {}).get('input')):
            msg = f'Input schema not found in the Actor build for Actor: {actor_id}'
            raise ValueError(msg)

        properties, required = _prune_actor_input_schema(actor_input)
        properties = {'run_input': properties}

        description = (
            'JSON encoded as a string with input schema '
            '(STRICTLY FOLLOW JSON FORMAT AND SCHEMA):\n\n'
            f'{json.dumps(properties, separators=(",", ":"))}'
            '\nIF THE TOOL INPUT SCHEMA SUPPORTS IT LIMIT THE NUMBER OF RESULTS'
        )
        if required:
            description += '\n\nRequired fields:\n' + '\n'.join(required)

        return create_model(
            'ApifyActorsToolInput',
            run_input=(str | dict, Field(..., description=description)),
        )

    def _run_actor(self, run_input: dict) -> list[dict]:
        """Run an Apify Actor and return the output dataset.

        Args:
            run_input: dict, JSON input for the Actor

        Returns:
            list[dict]: The output dataset

        Raises:
            ValueError: If the Actor was not started properly or the Run ID was not found in the run details
        """
        if (details := self._apify_client.actor(actor_id=self._actor_id).call(run_input=run_input)) is None:
            msg = f'Actor: {self._actor_id} was not started properly and details about the run were not returned'
            raise ValueError(msg)
        if (run_id := details.get('id')) is None:
            msg = f'Run ID not found in the run details for Actor: {self._actor_id}'
            raise ValueError(msg)
        run = self._apify_client.run(run_id=run_id)

        return run.dataset().list_items(clean=True).items
