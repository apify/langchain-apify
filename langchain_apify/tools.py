from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, create_model

from langchain_apify._client import ApifyToolsClient
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _MAX_DESCRIPTION_LEN,
    _actor_id_to_tool_name,
    _create_apify_client,
    _get_actor_latest_build,
    _prune_actor_input_schema,
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

    _apify_client: ApifyClient = PrivateAttr()
    _actor_id: str = PrivateAttr()

    def __init__(
        self,
        actor_id: str,
        apify_api_token: str | SecretStr | None = None,
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
        _raw_token: str | None = (
            apify_api_token.get_secret_value()
            if isinstance(apify_api_token, SecretStr)
            else apify_api_token or os.getenv('APIFY_API_TOKEN')
        )
        if not _raw_token:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)

        apify_client = _create_apify_client(ApifyClient, _raw_token)

        kwargs.update(
            {
                'name': _actor_id_to_tool_name(actor_id),
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
        build = _get_actor_latest_build(apify_client, actor_id)
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
        build = _get_actor_latest_build(apify_client, actor_id)
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


class ApifyGoogleSearchInput(BaseModel):
    """Input schema for :class:`ApifyGoogleSearchTool`."""

    query: str = Field(description='Search query string.')
    max_results: int = Field(default=10, description='Maximum number of search results to return.')
    country_code: str | None = Field(default=None, description='Two-letter country code for localised results.')
    language_code: str | None = Field(default=None, description='Two-letter language code.')


class ApifyWebCrawlerInput(BaseModel):
    """Input schema for :class:`ApifyWebCrawlerTool`."""

    url: str = Field(description='Seed URL to start crawling from.')
    max_crawl_pages: int = Field(default=10, description='Maximum number of pages to crawl.')
    max_crawl_depth: int = Field(default=1, description='Maximum link-follow depth from the seed URL.')
    crawler_type: str = Field(default='cheerio', description='Crawler engine (e.g. "cheerio", "playwright").')
    timeout_secs: int = Field(default=300, description='Maximum time in seconds to wait for the crawl to finish.')


class ApifyRunTaskInput(BaseModel):
    """Input schema for :class:`ApifyRunTaskTool`."""

    task_id: str = Field(description='Task ID or name (e.g. "user/my-task").')
    task_input: dict | None = Field(
        default=None, description="JSON-serialisable input that overrides the task's pre-saved input."
    )
    timeout_secs: int = Field(default=300, description='Maximum time in seconds to wait for the run to finish.')
    memory_mbytes: int | None = Field(
        default=None, description='Memory limit in MB for the run, or null for task default.'
    )


class ApifyRunTaskAndGetItemsInput(BaseModel):
    """Input schema for :class:`ApifyRunTaskAndGetItemsTool`."""

    task_id: str = Field(description='Task ID or name (e.g. "user/my-task").')
    task_input: dict | None = Field(
        default=None, description="JSON-serialisable input that overrides the task's pre-saved input."
    )
    timeout_secs: int = Field(default=300, description='Maximum time in seconds to wait for the run to finish.')
    memory_mbytes: int | None = Field(
        default=None, description='Memory limit in MB for the run, or null for task default.'
    )
    dataset_items_limit: int = Field(default=100, description='Maximum number of dataset items to return.')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(value: str | datetime | None) -> str | None:
    """Coerce a possible ``datetime`` to an ISO-8601 string."""
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
# Shared base for generic tools
# ---------------------------------------------------------------------------


class _ApifyGenericTool(BaseTool):  # type: ignore[override]
    """Shared base for all generic Apify tools.

    Handles ``ApifyToolsClient`` creation, sets ``handle_tool_error``,
    and defines developer-controlled safety limits that clamp values the
    LLM may provide at invocation time.

    Subclasses only need to declare ``name``, ``description``,
    ``args_schema``, and ``_run()``.
    """

    handle_tool_error: bool = True

    apify_api_token: SecretStr | None = Field(
        default_factory=secret_from_env('APIFY_API_TOKEN', default=None),
        description='Apify API token. Falls back to the APIFY_API_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    max_timeout_secs: int = Field(default=600, description='Upper bound for timeout_secs the LLM may request.')
    max_memory_mbytes: int = Field(default=32768, description='Upper bound for memory_mbytes the LLM may request.')
    max_items: int = Field(default=1000, description='Upper bound for limit / dataset_items_limit the LLM may request.')

    _client: ApifyToolsClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if self.apify_api_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._client = ApifyToolsClient(apify_api_token=self.apify_api_token.get_secret_value())
        super().model_post_init(__context)

    def _clamp_timeout(self, value: int) -> int:
        return max(1, min(value, self.max_timeout_secs))

    def _clamp_memory(self, value: int | None) -> int | None:
        if value is None:
            return None
        return max(1, min(value, self.max_memory_mbytes))

    def _clamp_items(self, value: int) -> int:
        return max(1, min(value, self.max_items))


# ---------------------------------------------------------------------------
# Generic tools
# ---------------------------------------------------------------------------


class ApifyRunActorTool(_ApifyGenericTool):  # type: ignore[override]
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

    def _run(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = 300,
        memory_mbytes: int | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run = self._client.run_actor(
                actor_id, run_input, self._clamp_timeout(timeout_secs), self._clamp_memory(memory_mbytes)
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps(_run_meta(run))


class ApifyGetDatasetItemsTool(_ApifyGenericTool):  # type: ignore[override]
    """Fetch items from an existing Apify dataset by ID.

    Returns a JSON object with an ``"items"`` key containing the list of item
    dicts.  When the dataset is empty an additional ``"message"`` key is
    included.

    Args:
        apify_api_token: Apify API token. Falls back to the ``APIFY_API_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"items": [...]}``; includes ``"message"`` when empty.

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
        'Fetch items from an Apify dataset by ID. Returns a JSON object with an "items" array.'
        ' Required: dataset_id (str) — Apify dataset ID.'
        ' Optional: limit (int, default 100), offset (int, default 0).'
    )
    args_schema: type[BaseModel] = ApifyGetDatasetItemsInput

    def _run(
        self,
        dataset_id: str,
        limit: int = 100,
        offset: int = 0,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            items = self._client.get_dataset_items(dataset_id, self._clamp_items(limit), offset)
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        if not items:
            return json.dumps({'items': [], 'message': 'Dataset is empty or not found.'})
        return json.dumps({'items': items})


class ApifyRunActorAndGetItemsTool(_ApifyGenericTool):  # type: ignore[override]
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
                actor_id,
                run_input,
                self._clamp_timeout(timeout_secs),
                self._clamp_memory(memory_mbytes),
                self._clamp_items(dataset_items_limit),
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})


class ApifyScrapeUrlTool(_ApifyGenericTool):  # type: ignore[override]
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

    def _run(
        self,
        url: str,
        timeout_secs: int = 120,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return self._client.scrape_url(url, self._clamp_timeout(timeout_secs))
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc


class ApifyRunTaskTool(_ApifyGenericTool):  # type: ignore[override]
    """Run a saved Apify Actor task by ID and return run metadata.

    Actor tasks are pre-configured Actor runs saved in the Apify Console.
    This tool starts a task with optional input overrides and returns run
    metadata (run ID, status, dataset ID, timestamps) as a JSON string.
    Use :class:`ApifyGetDatasetItemsTool` afterwards to retrieve results.

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

            from langchain_apify import ApifyRunTaskTool

            tool = ApifyRunTaskTool()
            result = tool.invoke({
                "task_id": "user/my-task",
                "task_input": {"key": "value"},
            })
    """

    name: str = 'apify_run_task'
    description: str = (
        'Run a saved Apify Actor task synchronously and return run metadata as a JSON string.'
        ' Required: task_id (str) — task ID or name (e.g. "user/my-task").'
        ' Optional: task_input (dict), timeout_secs (int, default 300),'
        ' memory_mbytes (int|null).'
        ' Returns JSON with keys: run_id, status, dataset_id, started_at, finished_at.'
        ' Use apify_get_dataset_items with the returned dataset_id to fetch results.'
    )
    args_schema: type[BaseModel] = ApifyRunTaskInput

    def _run(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = 300,
        memory_mbytes: int | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run = self._client.run_task(
                task_id, task_input, self._clamp_timeout(timeout_secs), self._clamp_memory(memory_mbytes)
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps(_run_meta(run))


class ApifyRunTaskAndGetItemsTool(_ApifyGenericTool):  # type: ignore[override]
    """Run a saved Apify Actor task and return both run metadata and dataset items.

    Combines :class:`ApifyRunTaskTool` and :class:`ApifyGetDatasetItemsTool`
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

            from langchain_apify import ApifyRunTaskAndGetItemsTool

            tool = ApifyRunTaskAndGetItemsTool()
            result = tool.invoke({
                "task_id": "user/my-task",
                "task_input": {"key": "value"},
            })
    """

    name: str = 'apify_run_task_and_get_items'
    description: str = (
        'Run a saved Apify Actor task synchronously and return both run metadata and dataset items.'
        ' Required: task_id (str) — task ID or name (e.g. "user/my-task").'
        ' Optional: task_input (dict), timeout_secs (int, default 300),'
        ' memory_mbytes (int|null), dataset_items_limit (int, default 100).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at)'
        ' and items (list of dataset item dicts).'
    )
    args_schema: type[BaseModel] = ApifyRunTaskAndGetItemsInput

    def _run(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = 300,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = 100,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run, items = self._client.run_task_and_get_items(
                task_id,
                task_input,
                self._clamp_timeout(timeout_secs),
                self._clamp_memory(memory_mbytes),
                self._clamp_items(dataset_items_limit),
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return json.dumps({'run': _run_meta(run), 'items': items})
