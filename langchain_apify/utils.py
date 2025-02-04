import string
from typing import Type, TypeVar

from apify_client import ApifyClientAsync
from apify_client.client import ApifyClient


def prune_actor_input_schema(
    input_schema: dict, max_description_len: int = 250
) -> tuple[dict, list[str]]:
    """Get the input schema from the Actor build.

    Trim the description to 250 characters.
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

    :param t: ApifyClient or ApifyClientAsync
    :param token: API token
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
    """
    valid_chars = string.ascii_letters + string.digits + "_-"
    return "apify_actor_" + "".join(
        char if char in valid_chars else "_" for char in actor_id
    )


def get_actor_latest_build(client: ApifyClient, actor_id: str) -> dict:
    """Get the latest build of an Actor from default build tag."""
    actor = client.actor(actor_id=actor_id)
    if not (actor_info := actor.get()):
        msg = f"Actor {actor_id} not found."
        raise ValueError(msg)

    default_build_tag = actor_info.get("defaultRunOptions", {}).get("build")
    latest_build_id = (
        actor_info.get("taggedBuilds", {}).get(default_build_tag, {}).get("buildId")
    )

    if (build := client.build(latest_build_id).get()) is None:
        msg = f"Build {latest_build_id} not found."
        raise ValueError(msg)

    return build
