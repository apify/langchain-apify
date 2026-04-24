from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from apify_client import ApifyClient, ApifyClientAsync
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import _create_apify_client
from langchain_apify.document_loaders import ApifyDatasetLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.documents import Document


class ApifyWrapper(BaseModel):
    """Wrapper around Apify client for LangChain.

    To use, you should have the environment variable `APIFY_API_TOKEN` set
    with your API key, or pass `apify_api_token`
    as a named parameter to the constructor.

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

    apify_api_token: SecretStr | None = Field(
        default_factory=secret_from_env('APIFY_API_TOKEN', default=None),
        description='Apify API token. Falls back to the APIFY_API_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    apify_client: ApifyClient = Field(default=None, exclude=True)  # type: ignore[assignment]
    apify_client_async: ApifyClientAsync = Field(default=None, exclude=True)  # type: ignore[assignment]

    def __init__(
        self,
        apify_api_token: str | SecretStr | None = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialise the wrapper.

        Args:
            apify_api_token (Optional[str | SecretStr]): Apify API token. Falls
                back to the ``APIFY_API_TOKEN`` environment variable when *None*.
            *args: Any: Additional positional arguments forwarded to Pydantic.
            **kwargs: Any: Additional keyword arguments forwarded to Pydantic.
        """
        # Only forward the token when explicitly provided; otherwise let the
        # Pydantic ``default_factory`` read it from the environment.
        if apify_api_token is not None:
            kwargs['apify_api_token'] = apify_api_token
        super().__init__(*args, **kwargs)

    @model_validator(mode='after')
    def _init_clients(self) -> ApifyWrapper:
        """Validate the token and initialise both sync and async Apify clients.

        Returns:
            ApifyWrapper: The validated wrapper instance.

        Raises:
            ValueError: If no token is provided and APIFY_API_TOKEN is not set.
        """
        if self.apify_api_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        token = self.apify_api_token.get_secret_value()
        self.apify_client = _create_apify_client(ApifyClient, token)
        self.apify_client_async = _create_apify_client(ApifyClientAsync, token)
        return self

    def call_actor(  # noqa: PLR0913
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
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

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.

        Raises:
            RuntimeError: If the Actor call fails.
        """
        if (
            actor_call := self.apify_client.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f'Failed to call Actor {actor_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call['defaultDatasetId'],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor(  # noqa: PLR0913
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
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

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.

        Raises:
            RuntimeError: If the Actor call fails.
        """
        if (
            actor_call := await self.apify_client_async.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f'Failed to call Actor {actor_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call['defaultDatasetId'],
            dataset_mapping_function=dataset_mapping_function,
        )

    def call_actor_task(  # noqa: PLR0913
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
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

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.

        Raises:
            RuntimeError: If the task call fails.
        """
        if (
            task_call := self.apify_client.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f'Failed to call task {task_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call['defaultDatasetId'],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor_task(  # noqa: PLR0913
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
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

        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.

        Raises:
            RuntimeError: If the task call fails.
        """
        if (
            task_call := await self.apify_client_async.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f'Failed to call task {task_id}.'
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call['defaultDatasetId'],
            dataset_mapping_function=dataset_mapping_function,
        )
