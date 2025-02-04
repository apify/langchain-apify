from typing import Any, Callable, Optional

from apify_client import ApifyClient, ApifyClientAsync
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

from langchain_apify.document_loaders import ApifyDatasetLoader
from langchain_apify.utils import create_apify_client


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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    apify_client: ApifyClient
    apify_client_async: ApifyClientAsync
    apify_api_token: Optional[str] = None

    def __init__(
        self,
        apify_api_token: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the loader with an Apify dataset ID and a mapping function.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an instance
                of the Document class.
        """
        kwargs.update({"apify_api_token": apify_api_token})
        super().__init__(*args, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate environment.
        Validate that an Apify API token is set and the apify-client
        Python package exists in the current environment.
        """
        apify_api_token = get_from_dict_or_env(
            values, "apify_api_token", "APIFY_API_TOKEN"
        )

        client = create_apify_client(ApifyClient, apify_api_token)
        async_client = create_apify_client(ApifyClientAsync, apify_api_token)

        values["apify_client"] = client
        values["apify_client_async"] = async_client

        return values

    def call_actor(
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
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
        """
        if (
            actor_call := self.apify_client.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f"Failed to call actor {actor_id}."
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor(
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
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
        """
        if (
            actor_call := await self.apify_client_async.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f"Failed to call actor {actor_id}."
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=actor_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    def call_actor_task(
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
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
        """
        if (
            task_call := self.apify_client.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f"Failed to call task {task_id}."
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor_task(
        self,
        task_id: str,
        task_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
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
        """
        if (
            task_call := await self.apify_client_async.task(task_id).call(
                task_input=task_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ) is None:
            msg = f"Failed to call task {task_id}."
            raise RuntimeError(msg)

        return ApifyDatasetLoader(
            dataset_id=task_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )
