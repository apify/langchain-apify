"""Core generic Apify tools and their input schemas.

These tools wrap general Apify platform primitives (running Actors and tasks,
fetching dataset items, scraping a single URL) behind LLM-friendly interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import ArgsSchema, ToolException
from pydantic import BaseModel, Field

from langchain_apify._constants import (
    _DEFAULT_DATASET_ITEMS_LIMIT,
    _DEFAULT_RUN_TIMEOUT_SECS,
    _DEFAULT_SCRAPE_TIMEOUT_SECS,
    _MAX_ITEMS_CAP,
    _MAX_MEMORY_MBYTES_CAP,
    _MAX_TIMEOUT_SECS_CAP,
)
from langchain_apify.tools.base import _ApifyGenericTool

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Input schemas for the generic tools
# ---------------------------------------------------------------------------

_DESC_RUN_TIMEOUT_SECS = (
    f'Maximum time in seconds to wait for the run to finish (clamped to {_MAX_TIMEOUT_SECS_CAP} max).'
)
_DESC_MEMORY_MBYTES = (
    f'Memory per run in MB. Power of 2 from 128, or null for default (clamped to {_MAX_MEMORY_MBYTES_CAP} max).'
)
_DESC_DATASET_ITEMS_LIMIT = f'Maximum number of dataset items to return (clamped to {_MAX_ITEMS_CAP} max).'


class ApifyRunActorInput(BaseModel):
    """Input schema for :class:`ApifyRunActorTool`."""

    actor_id: str = Field(description='Actor ID or name (e.g. "apify/python-example").')
    run_input: dict | None = Field(default=None, description='JSON-serialisable input for the Actor.')
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)
    memory_mbytes: int | None = Field(default=None, description=_DESC_MEMORY_MBYTES)


class ApifyGetDatasetItemsInput(BaseModel):
    """Input schema for :class:`ApifyGetDatasetItemsTool`."""

    dataset_id: str = Field(description='Apify dataset ID.')
    limit: int = Field(
        default=_DEFAULT_DATASET_ITEMS_LIMIT,
        description=f'Maximum number of items to return (clamped to {_MAX_ITEMS_CAP} max).',
    )
    offset: int = Field(default=0, description='Number of items to skip from the start.')


class ApifyRunActorAndGetDatasetInput(BaseModel):
    """Input schema for :class:`ApifyRunActorAndGetDatasetTool`."""

    actor_id: str = Field(description='Actor ID or name (e.g. "apify/python-example").')
    run_input: dict | None = Field(default=None, description='JSON-serialisable input for the Actor.')
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)
    memory_mbytes: int | None = Field(default=None, description=_DESC_MEMORY_MBYTES)
    dataset_items_limit: int = Field(default=_DEFAULT_DATASET_ITEMS_LIMIT, description=_DESC_DATASET_ITEMS_LIMIT)


class ApifyScrapeUrlInput(BaseModel):
    """Input schema for :class:`ApifyScrapeUrlTool`."""

    url: str = Field(description='The URL to scrape.')
    timeout_secs: int = Field(
        default=_DEFAULT_SCRAPE_TIMEOUT_SECS,
        description=(
            f'Maximum time in seconds to wait for the crawl to finish (clamped to {_MAX_TIMEOUT_SECS_CAP} max).'
        ),
    )


class ApifyRunTaskInput(BaseModel):
    """Input schema for :class:`ApifyRunTaskTool`."""

    task_id: str = Field(description='Task ID or name (e.g. "user/my-task").')
    task_input: dict | None = Field(
        default=None, description="JSON-serialisable input that overrides the task's pre-saved input."
    )
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)
    memory_mbytes: int | None = Field(default=None, description=_DESC_MEMORY_MBYTES)


class ApifyRunTaskAndGetDatasetInput(BaseModel):
    """Input schema for :class:`ApifyRunTaskAndGetDatasetTool`."""

    task_id: str = Field(description='Task ID or name (e.g. "user/my-task").')
    task_input: dict | None = Field(
        default=None, description="JSON-serialisable input that overrides the task's pre-saved input."
    )
    timeout_secs: int = Field(default=_DEFAULT_RUN_TIMEOUT_SECS, description=_DESC_RUN_TIMEOUT_SECS)
    memory_mbytes: int | None = Field(default=None, description=_DESC_MEMORY_MBYTES)
    dataset_items_limit: int = Field(default=_DEFAULT_DATASET_ITEMS_LIMIT, description=_DESC_DATASET_ITEMS_LIMIT)


# ---------------------------------------------------------------------------
# Generic tools
# ---------------------------------------------------------------------------


class ApifyRunActorTool(_ApifyGenericTool):  # type: ignore[override]
    """Run any Apify Actor by ID with an arbitrary JSON input.

    Returns run metadata in a JSON envelope.  Use
    :class:`ApifyGetDatasetItemsTool` afterwards to retrieve the results from
    the dataset.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": []}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyRunActorTool

            tool = ApifyRunActorTool()
            result = tool.invoke({
                "actor_id": "apify/python-example",
                "run_input": {"first_number": 2, "second_number": 3},
            })
    """

    name: str = 'apify_run_actor'
    description: str = (
        'Run an Apify Actor synchronously and return a JSON envelope.'
        ' Required: actor_id (str); Actor ID or name (e.g. "apify/python-example").'
        f' Optional: run_input (dict), timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}),'
        ' memory_mbytes (int|null).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use apify_get_dataset_items with run.dataset_id to fetch results.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyRunActorInput

    def _run(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run = self._client.run_actor(
                actor_id, run_input, self._clamp_timeout(timeout_secs), self._clamp_memory(memory_mbytes)
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, [])


class ApifyGetDatasetItemsTool(_ApifyGenericTool):  # type: ignore[override]
    """Fetch items from an existing Apify dataset by ID.

    Returns a JSON object with ``"run"`` (always ``null`` here, since no Actor
    is run) and ``"items"`` (the list of item dicts, empty when the dataset
    has no items).

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": null, "items": [...]}``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyGetDatasetItemsTool

            tool = ApifyGetDatasetItemsTool()
            result = tool.invoke({"dataset_id": "abc123", "limit": 10})
    """

    name: str = 'apify_get_dataset_items'
    description: str = (
        'Fetch items from an Apify dataset by ID and return a JSON envelope.'
        ' Required: dataset_id (str); Apify dataset ID.'
        f' Optional: limit (int, default {_DEFAULT_DATASET_ITEMS_LIMIT}), offset (int, default 0).'
        ' Returns JSON with keys: run (null), items (empty array when the dataset has no items).'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyGetDatasetItemsInput

    def _run(
        self,
        dataset_id: str,
        limit: int = _DEFAULT_DATASET_ITEMS_LIMIT,
        offset: int = 0,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            items = self._client.get_dataset_items(dataset_id, self._clamp_items(limit), max(0, offset))
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(None, items)


class ApifyRunActorAndGetDatasetTool(_ApifyGenericTool):  # type: ignore[override]
    """Run any Apify Actor and return both run metadata and dataset items.

    Combines :class:`ApifyRunActorTool` and :class:`ApifyGetDatasetItemsTool`
    into a single call.  Returns a JSON envelope.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [...]}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``
        and ``items`` are the dataset item dicts.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyRunActorAndGetDatasetTool

            tool = ApifyRunActorAndGetDatasetTool()
            result = tool.invoke({
                "actor_id": "apify/python-example",
                "run_input": {"first_number": 2, "second_number": 3},
            })
    """

    name: str = 'apify_run_actor_and_get_dataset'
    description: str = (
        'Run an Apify Actor synchronously and return a JSON envelope.'
        ' Required: actor_id (str); Actor ID or name (e.g. "apify/python-example").'
        f' Optional: run_input (dict), timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}),'
        f' memory_mbytes (int|null), dataset_items_limit (int, default {_DEFAULT_DATASET_ITEMS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at)'
        ' and items (list of dataset item dicts).'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyRunActorAndGetDatasetInput

    def _run(
        self,
        actor_id: str,
        run_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = _DEFAULT_DATASET_ITEMS_LIMIT,
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
        return self._envelope(run, items)


class ApifyScrapeUrlTool(_ApifyGenericTool):  # type: ignore[override]
    """Scrape a single URL and return its content in a JSON envelope.

    Uses the ``apify/website-content-crawler`` Actor under the hood with
    ``maxCrawlPages=1``.  The scraped content (markdown, or plain text when
    markdown is unavailable) is the ``content`` field of the single item.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [{"url": ..., "content": ...}]}``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyScrapeUrlTool

            tool = ApifyScrapeUrlTool()
            markdown = tool.invoke({"url": "https://apify.com"})
    """

    name: str = 'apify_scrape_url'
    description: str = (
        'Scrape a single URL using Apify and return a JSON envelope.'
        ' Required: url (str); the URL to scrape.'
        f' Optional: timeout_secs (int, default {_DEFAULT_SCRAPE_TIMEOUT_SECS}).'
        ' Returns JSON with keys: run, items ([{url, content}];'
        ' content is markdown, or plain text when markdown is unavailable).'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyScrapeUrlInput

    def _run(
        self,
        url: str,
        timeout_secs: int = _DEFAULT_SCRAPE_TIMEOUT_SECS,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            # _scrape_url is the rich primitive; scrape_url() drops the metadata
            # this tool needs (run + content source), so access it directly.
            run, _, content, _ = self._client._scrape_url(url, self._clamp_timeout(timeout_secs))  # noqa: SLF001
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, [{'url': url, 'content': content}])


class ApifyRunTaskTool(_ApifyGenericTool):  # type: ignore[override]
    """Run a saved Apify Actor task by ID and return run metadata.

    Actor tasks are pre-configured Actor runs saved in the Apify Console.
    This tool starts a task with optional input overrides and returns run
    metadata in a JSON envelope.  Use :class:`ApifyGetDatasetItemsTool`
    afterwards to retrieve results.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": []}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyRunTaskTool

            tool = ApifyRunTaskTool()
            result = tool.invoke({
                "task_id": "user/my-task",
                "task_input": {"key": "value"},
            })
    """

    name: str = 'apify_run_task'
    description: str = (
        'Run a saved Apify Actor task synchronously and return a JSON envelope.'
        ' Required: task_id (str); task ID or name (e.g. "user/my-task").'
        f' Optional: task_input (dict), timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}),'
        ' memory_mbytes (int|null).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at), items.'
        ' Use apify_get_dataset_items with run.dataset_id to fetch results.'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyRunTaskInput

    def _run(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            run = self._client.run_task(
                task_id, task_input, self._clamp_timeout(timeout_secs), self._clamp_memory(memory_mbytes)
            )
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return self._envelope(run, [])


class ApifyRunTaskAndGetDatasetTool(_ApifyGenericTool):  # type: ignore[override]
    """Run a saved Apify Actor task and return both run metadata and dataset items.

    Combines :class:`ApifyRunTaskTool` and :class:`ApifyGetDatasetItemsTool`
    into a single call.  Returns a JSON envelope.

    Args:
        apify_token: Apify API token. Falls back to the ``APIFY_TOKEN``
            environment variable when *None*.

    Returns:
        JSON object ``{"run": {...}, "items": [...]}`` where ``run`` holds
        ``run_id``, ``status``, ``dataset_id``, ``started_at``, ``finished_at``
        and ``items`` are the dataset item dicts.

    Example:
        .. code-block:: python

            import os
            os.environ["APIFY_TOKEN"] = "your-apify-token"

            from langchain_apify import ApifyRunTaskAndGetDatasetTool

            tool = ApifyRunTaskAndGetDatasetTool()
            result = tool.invoke({
                "task_id": "user/my-task",
                "task_input": {"key": "value"},
            })
    """

    name: str = 'apify_run_task_and_get_dataset'
    description: str = (
        'Run a saved Apify Actor task synchronously and return a JSON envelope.'
        ' Required: task_id (str); task ID or name (e.g. "user/my-task").'
        f' Optional: task_input (dict), timeout_secs (int, default {_DEFAULT_RUN_TIMEOUT_SECS}),'
        f' memory_mbytes (int|null), dataset_items_limit (int, default {_DEFAULT_DATASET_ITEMS_LIMIT}).'
        ' Returns JSON with keys: run (run_id, status, dataset_id, started_at, finished_at)'
        ' and items (list of dataset item dicts).'
        ' Use only the data returned; do not hallucinate missing fields.'
    )
    args_schema: ArgsSchema | None = ApifyRunTaskAndGetDatasetInput

    def _run(
        self,
        task_id: str,
        task_input: dict | None = None,
        timeout_secs: int = _DEFAULT_RUN_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        dataset_items_limit: int = _DEFAULT_DATASET_ITEMS_LIMIT,
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
        return self._envelope(run, items)


# Convenience tool-class list for selective agent binding.
APIFY_CORE_TOOLS: list[type[BaseTool]] = [
    ApifyRunActorTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyScrapeUrlTool,
    ApifyRunTaskTool,
    ApifyRunTaskAndGetDatasetTool,
]
