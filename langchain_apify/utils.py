import string
from typing import Type, TypeVar

import requests
from apify_client import ApifyClientAsync
from apify_client.client import ApifyClient


def prune_actor_input_schema(
    input_schema: dict, max_description_len: int = 250
) -> tuple[dict, list[str]]:
    """Get the input schema from the Actor build.

    Trim the description to 250 characters.

    Args:
        input_schema (dict): The input schema from the Actor build.
        max_description_len (int): The maximum length of the description.

    Returns:
        tuple[dict, list[str]]: A tuple containing the pruned properties
            and required fields.
    """
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    properties_out: dict = {}
    for item, meta in properties.items():
        properties_out[item] = {}
        if desc := meta.get("description"):
            properties_out[item]["description"] = (
                desc[:max_description_len] + "..."
                if len(desc) > max_description_len
                else desc
            )
        for key_name in ("type", "default", "prefill", "enum"):
            if value := meta.get(key_name):
                properties_out[item][key_name] = value

    return properties_out, required


T = TypeVar("T", ApifyClient, ApifyClientAsync)


def create_apify_client(t: Type[T], token: str) -> T:
    """Create an Apify client instance with custom user-agent.

    Args:
        t (Type[T]): ApifyClient or ApifyClientAsync.
        token (str): API token.

    Returns:
        T: ApifyClient or ApifyClientAsync instance.
    """
    if not token:
        msg = "API token is required."
        raise ValueError(msg)
    client = t(token)
    http_client_attr = (
        "httpx_async_client" if isinstance(client, ApifyClientAsync) else "httpx_client"
    )
    if http_client := getattr(client.http_client, http_client_attr):
        http_client.headers["user-agent"] += "; Origin/langchain"
    return client


def actor_id_to_tool_name(actor_id: str) -> str:
    """Turn actor_id into a valid tool name.

    Tool name must only contain letters, numbers, underscores, dashes,
    and cannot contain spaces.

    Args:
        actor_id (str): Actor ID from Apify store.

    Returns:
        str: A valid tool name.
    """
    valid_chars = string.ascii_letters + string.digits + "_-"
    return "apify_actor_" + "".join(
        char if char in valid_chars else "_" for char in actor_id
    )


def get_actor_latest_build(apify_client: ApifyClient, actor_id: str) -> dict:
    """Get the latest build of an Actor from default build tag.

    Args:
        client (ApifyClient): An instance of the ApifyClient class.
        actor_id (str): Actor name from Apify store to run.

    Returns:
        dict: The latest build of the Actor.
    """
    if (actor := apify_client.actor(actor_id).get()) is None:
        msg = f"Actor {actor_id} not found."
        raise ValueError(msg)

    if (actor_obj_id := actor.get("id")) is None:
        msg = f"Failed to get the Actor object ID for {actor_id}."
        raise ValueError(msg)

    url = f"https://api.apify.com/v2/acts/{actor_obj_id}/builds/default"
    response = requests.request("GET", url)

    build = response.json()
    if not isinstance(build, dict):
        msg = f"Failed to get the latest build of the Actor {actor_id}."
        raise ValueError(msg)

    if (data := build.get("data")) is None:
        msg = f"Failed to get the latest build data of the Actor {actor_id}."
        raise ValueError(msg)

    return data
