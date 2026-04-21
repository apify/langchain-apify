from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from datetime import datetime

from apify_client import ApifyClient
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, create_model

from langchain_apify._client import ApifyToolsClient
from langchain_apify._error_messages import ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify.utils import (
    _MAX_DESCRIPTION_LEN,
    actor_id_to_tool_name,
    create_apify_client,
    get_actor_latest_build,
    prune_actor_input_schema,
)

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

        Raises:
            ValueError: If the `APIFY_API_TOKEN` environment variable is not set
        """
        apify_api_token = apify_api_token or os.getenv('APIFY_API_TOKEN')
        if not apify_api_token:
            msg = ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
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

        Returns:
            list[dict]: The output dataset.
        """
        input_dict = json.loads(run_input) if isinstance(run_input, str) else run_input
        # retrieve if nested, just in case
        input_dict = input_dict.get('run_input', input_dict)
        return self._run_actor(input_dict)

    @staticmethod
    def _create_description(apify_client: ApifyClient, actor_id: str) -> str:
        """Create a description for the tool.

        Args:
            apify_client (ApifyClient): Apify client instance.
            actor_id (str): Actor name from Apify store to run.

        Returns:
            str: The description.
        """
        build = get_actor_latest_build(apify_client, actor_id)
        actor_description = build.get('actorDefinition', {}).get('description', '')
        if len(actor_description) > _MAX_DESCRIPTION_LEN:
            actor_description = actor_description[:_MAX_DESCRIPTION_LEN] + '...(TRUNCATED, TOO LONG)'
        return actor_description

    @staticmethod
    def _build_tool_args_schema_model(
        apify_client: ApifyClient,
        actor_id: str,
    ) -> type[BaseModel]:
        """Build a tool class for an agent that runs the Apify Actor.

        Args:
            apify_client (ApifyClient): Apify client instance.
            actor_id (str): Actor name from Apify store to run.

        Returns:
            type[BaseModel]: The tool input model class for the Apify Actor.

        Raises:
            ValueError: If the input schema is not found in the Actor build.
        """
        build = get_actor_latest_build(apify_client, actor_id)
        if not (actor_input := build.get('actorDefinition', {}).get('input')):
            msg = f'Input schema not found in the Actor build for Actor: {actor_id}'
            raise ValueError(msg)

        properties, required = prune_actor_input_schema(actor_input)
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


# ---------------------------------------------------------------------------
# Input schemas for the generic tools
# ---------------------------------------------------------------------------


class ApifyRunActorInput(BaseModel):
    """Input schema for :class:`ApifyRunActorTool`."""

    actor_id: str = Field(description='Actor ID or name (e.g. "apify/python-example").')
    run_input: dict | None = Field(default=None, description='JSON-serialisable input for the Actor.')
    timeout_secs: int = Field(default=300, description='Maximum time in seconds to wait for the run to finish.')
    memory_mbytes: int | None = Field(default=None, description='Memory limit in MB for the run, or null for default.')


class ApifyGetDatasetItemsInput(BaseModel):
    """Input schema for :class:`ApifyGetDatasetItemsTool`."""

    dataset_id: str = Field(description='Apify dataset ID.')
    limit: int = Field(default=100, description='Maximum number of items to return.')
    offset: int = Field(default=0, description='Number of items to skip from the start.')


class ApifyRunActorAndGetItemsInput(BaseModel):
    """Input schema for :class:`ApifyRunActorAndGetItemsTool`."""

    actor_id: str = Field(description='Actor ID or name (e.g. "apify/python-example").')
    run_input: dict | None = Field(default=None, description='JSON-serialisable input for the Actor.')
    timeout_secs: int = Field(default=300, description='Maximum time in seconds to wait for the run to finish.')
    memory_mbytes: int | None = Field(default=None, description='Memory limit in MB for the run, or null for default.')
    dataset_items_limit: int = Field(default=100, description='Maximum number of dataset items to return.')


class ApifyScrapeUrlInput(BaseModel):
    """Input schema for :class:`ApifyScrapeUrlTool`."""

    url: str = Field(description='The URL to scrape.')
    timeout_secs: int = Field(default=120, description='Maximum time in seconds to wait for the crawl to finish.')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(value: str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _run_meta(run: dict) -> dict:
    """Extract a compact metadata dict from an Apify run-details dict."""
    return {
        'run_id': run.get('id'),
        'status': run.get('status'),
        'dataset_id': run.get('defaultDatasetId'),
        'started_at': _iso(run.get('startedAt')),
        'finished_at': _iso(run.get('finishedAt')),
    }


# ---------------------------------------------------------------------------
# Generic tools
# ---------------------------------------------------------------------------


class ApifyRunActorTool(BaseTool):  # type: ignore[override]
    """Run any Apify Actor by ID with an arbitrary JSON input.

    Returns run metadata (run ID, status, dataset ID, timestamps) as a JSON
    string.  Use :class:`ApifyGetDatasetItemsTool` afterwards to retrieve the
    results from the dataset.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with keys ``run_id``, ``status``, ``dataset_id``,
        ``started_at``, and ``finished_at``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyRunActorTool

            tool = ApifyRunActorTool()
            result = tool.invoke({
                "actor_id": "apify/python-example",
                "run_input": {"first_number": 2, "second_number": 3},
            })
    """

    name: str = 'apify_run_actor'
    description: str = (
        'Run an Apify Actor synchronously and return run metadata as a JSON string.'
        ' Required: actor_id (str) — Actor ID or name (e.g. "apify/python-example").'
        ' Optional: run_input (dict), timeout_secs (int, default 300),'
        ' memory_mbytes (int|null).'
        ' Returns JSON with keys: run_id, status, dataset_id, started_at, finished_at.'
        ' Use apify_get_dataset_items with the returned dataset_id to fetch results.'
    )
    args_schema: type[BaseModel] = ApifyRunActorInput
    handle_tool_error: bool = True

    _client: ApifyToolsClient

    def __init__(self, apify_api_token: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._client = ApifyToolsClient(apify_api_token=apify_api_token)

    def _run(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = 300,
        memory_mbytes: int | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run = self._client.run_actor(actor_id, run_input, timeout_secs, memory_mbytes)
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps(_run_meta(run))


class ApifyGetDatasetItemsTool(BaseTool):  # type: ignore[override]
    """Fetch items from an existing Apify dataset by ID.

    Returns items as a JSON string.  When the dataset is empty the tool returns
    an informative JSON message instead of raising an error.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON array of item dicts, or ``{"items": [], "message": "..."}`` when
        the dataset is empty.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyGetDatasetItemsTool

            tool = ApifyGetDatasetItemsTool()
            result = tool.invoke({"dataset_id": "abc123", "limit": 10})
    """

    name: str = 'apify_get_dataset_items'
    description: str = (
        'Fetch items from an Apify dataset by ID. Returns a JSON array of item dicts.'
        ' Required: dataset_id (str) — Apify dataset ID.'
        ' Optional: limit (int, default 100), offset (int, default 0).'
        ' Returns an empty JSON object with a message when the dataset is empty.'
    )
    args_schema: type[BaseModel] = ApifyGetDatasetItemsInput
    handle_tool_error: bool = True

    _client: ApifyToolsClient

    def __init__(self, apify_api_token: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._client = ApifyToolsClient(apify_api_token=apify_api_token)

    def _run(
        self,
        dataset_id: str,
        limit: int = 100,
        offset: int = 0,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        items = self._client.get_dataset_items(dataset_id, limit, offset)
        if not items:
            return json.dumps({'items': [], 'message': 'Dataset is empty or not found.'})
        return json.dumps(items)


class ApifyRunActorAndGetItemsTool(BaseTool):  # type: ignore[override]
    """Run any Apify Actor and return both run metadata and dataset items.

    Combines :class:`ApifyRunActorTool` and :class:`ApifyGetDatasetItemsTool`
    into a single call.  Returns a JSON string with ``run`` (metadata) and
    ``items`` (list of dicts) keys.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON string with two keys: ``run`` (dict with ``run_id``, ``status``,
        ``dataset_id``, ``started_at``, ``finished_at``) and ``items`` (list
        of dataset item dicts).

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyRunActorAndGetItemsTool

            tool = ApifyRunActorAndGetItemsTool()
            result = tool.invoke({
                "actor_id": "apify/python-example",
                "run_input": {"first_number": 2, "second_number": 3},
            })
    """

    name: str = 'apify_run_actor_and_get_items'
    description: str = (
        'Run an Apify Actor synchronously and return both run metadata and dataset items.'
        ' Required: actor_id (str) — Actor ID or name (e.g. "apify/python-example").'
        ' Optional: run_input (dict), timeout_secs (int, default 300),'
        ' memory_mbytes (int|null), dataset_items_limit (int, default 100).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at)'
        ' and items (list of dataset item dicts).'
    )
    args_schema: type[BaseModel] = ApifyRunActorAndGetItemsInput
    handle_tool_error: bool = True

    _client: ApifyToolsClient

    def __init__(self, apify_api_token: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._client = ApifyToolsClient(apify_api_token=apify_api_token)

    def _run(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = 300,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = 100,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.run_actor_and_get_items(
                actor_id, run_input, timeout_secs, memory_mbytes, dataset_items_limit
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyScrapeUrlTool(BaseTool):  # type: ignore[override]
    """Scrape a single URL and return its content as markdown.

    Uses the ``apify/website-content-crawler`` Actor under the hood with
    ``maxCrawlPages=1``.  Returns the page content as a plain markdown string
    (not JSON).

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        Markdown string with the full text content of the scraped page, or a
        plain-text fallback when markdown is unavailable.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"

            from langchain_apify import ApifyScrapeUrlTool

            tool = ApifyScrapeUrlTool()
            markdown = tool.invoke({"url": "https://apify.com"})
    """

    name: str = 'apify_scrape_url'
    description: str = (
        'Scrape a single URL using Apify and return its full content as a markdown string.'
        ' Required: url (str) — the URL to scrape.'
        ' Optional: timeout_secs (int, default 120).'
        ' Returns the page content as markdown (or plain text if markdown is unavailable).'
    )
    args_schema: type[BaseModel] = ApifyScrapeUrlInput
    handle_tool_error: bool = True

    _client: ApifyToolsClient

    def __init__(self, apify_api_token: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._client = ApifyToolsClient(apify_api_token=apify_api_token)

    def _run(
        self,
        url: str,
        timeout_secs: int = 120,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return self._client.scrape_url(url, timeout_secs)
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
