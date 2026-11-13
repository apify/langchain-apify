from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient, ApifyClientAsync
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _apify_token_secret_factory,
    _charge_limit,
    _create_apify_client,
    _resolve_deprecated_token,
    _run_timeout,
)
from langchain_apify.document_loaders import ApifyDatasetLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.documents import Document


class ApifyWrapper(BaseModel):
    """Wrapper around Apify client for LangChain.

    To use, you should have the environment variable ``APIFY_TOKEN`` set
    with your API key, or pass ``apify_token`` as a named parameter to the
    constructor.

    For details, see https://docs.apify.com/platform/integrations/langchain

    Example:
        .. code-block:: python

            from langchain_apify import ApifyWrapper
            from langchain_core.documents import Document

            apify = ApifyWrapper()

            loader = apify.call_actor(
                actor_id="apify/website-content-crawler",
                run_input={
                    "startUrls": [
                        {"url": "https://python.langchain.com/docs/introduction/"}
                    ],
                    "maxCrawlPages": 10,
                    "crawlerType": "cheerio"
                },
                dataset_mapping_function=lambda item: Document(
                    page_content=item["text"] or "",
                    metadata={"source": item["url"]}
                ),
            )
            documents = loader.load()
    """

    # allow arbitrary types in the model config for the apify client fields
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    apify_token: SecretStr | None = Field(
        default_factory=_apify_token_secret_factory,
        description='Apify API token. Falls back to the APIFY_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    # The `_init_clients` validator always sets both clients, so callers never see the `None` default.
    apify_client: ApifyClient = Field(default=None, exclude=True)  # ty: ignore[invalid-assignment]
    apify_client_async: ApifyClientAsync = Field(default=None, exclude=True)  # ty: ignore[invalid-assignment]

    def __init__(
        self,
        apify_token: str | SecretStr | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the wrapper.

        Args:
            apify_token (Optional[str | SecretStr]): Apify API token. Falls
                back to the ``APIFY_TOKEN`` environment variable when *None*.
            apify_api_token: Deprecated alias for ``apify_token``.
            *args: Any: Additional positional arguments forwarded to Pydantic.
            **kwargs: Any: Additional keyword arguments forwarded to Pydantic.
        """
        if 'apify_api_token' in kwargs:
            apify_token = _resolve_deprecated_token(apify_token, kwargs.pop('apify_api_token'))

        if apify_token is not None:
            kwargs['apify_token'] = apify_token
        super().__init__(*args, **kwargs)

    @model_validator(mode='after')
    def _init_clients(self) -> ApifyWrapper:
        """Validate the token and initialise both sync and async Apify clients.

        Returns:
            ApifyWrapper: The validated wrapper instance.

        Raises:
            ValueError: If no token is provided and APIFY_TOKEN is not set.
        """
        if self.apify_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        token = self.apify_token.get_secret_value()
        self.apify_client = _create_apify_client(ApifyClient, token)
        self.apify_client_async = _create_apify_client(ApifyClientAsync, token)
        return self

    def call_actor(
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> ApifyDatasetLoader:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): Optional maximum total cost of the run, in USD.
                A limit below the Actor's own minimum makes the run fail.

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.

        Raises:
            RuntimeError: If the Actor call fails.
        """
        run_timeout = _run_timeout(timeout_secs)
        if (
            actor_call := self.apify_client.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                run_timeout=run_timeout,
                max_total_charge_usd=_charge_limit(max_total_charge_usd),
            )
        ) is None:
            msg = f'Failed to call Actor {actor_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call.default_dataset_id,
            dataset_mapping_function=dataset_mapping_function,
            apify_token=self.apify_token,
        )

    async def acall_actor(
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> ApifyDatasetLoader:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to
                an instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): Optional maximum total cost of the run, in USD.
                A limit below the Actor's own minimum makes the run fail.

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.

        Raises:
            RuntimeError: If the Actor call fails.
        """
        run_timeout = _run_timeout(timeout_secs)
        if (
            actor_call := await self.apify_client_async.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                run_timeout=run_timeout,
                max_total_charge_usd=_charge_limit(max_total_charge_usd),
            )
        ) is None:
            msg = f'Failed to call Actor {actor_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call.default_dataset_id,
            dataset_mapping_function=dataset_mapping_function,
            apify_token=self.apify_token,
        )

    def call_actor_task(
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> ApifyDatasetLoader:
        """Run a saved Actor task on Apify and wait for results to be ready.

        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            task_input (Dict): The input object of the task that you're trying to run.
                Overrides the task's saved input.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): Optional maximum total cost of the run, in USD.
                A limit below the Actor's own minimum makes the run fail.

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.

        Raises:
            RuntimeError: If the task call fails.
        """
        run_timeout = _run_timeout(timeout_secs)
        if (
            task_call := self.apify_client.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                run_timeout=run_timeout,
                max_total_charge_usd=_charge_limit(max_total_charge_usd),
            )
        ) is None:
            msg = f'Failed to call task {task_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call.default_dataset_id,
            dataset_mapping_function=dataset_mapping_function,
            apify_token=self.apify_token,
        )

    async def acall_actor_task(
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> ApifyDatasetLoader:
        """Run a saved Actor task on Apify and wait for results to be ready.

        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            task_input (Dict): The input object of the task that you're trying to run.
                Overrides the task's saved input.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): Optional maximum total cost of the run, in USD.
                A limit below the Actor's own minimum makes the run fail.

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.

        Raises:
            RuntimeError: If the task call fails.
        """
        run_timeout = _run_timeout(timeout_secs)
        if (
            task_call := await self.apify_client_async.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                run_timeout=run_timeout,
                max_total_charge_usd=_charge_limit(max_total_charge_usd),
            )
        ) is None:
            msg = f'Failed to call task {task_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call.default_dataset_id,
            dataset_mapping_function=dataset_mapping_function,
            apify_token=self.apify_token,
        )
