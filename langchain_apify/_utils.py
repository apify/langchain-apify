from __future__ import annotations

import string
from datetime import datetime
from typing import TypeVar

import requests
from apify_client import ApifyClientAsync
from apify_client.client import ApifyClient

_MAX_DESCRIPTION_LEN: int = 350
_REQUESTS_TIMEOUT_SECS: float = 10.0
_APIFY_API_ENDPOINT_GET_DEFAULT_BUILD: str = 'https://api.apify.com/v2/acts/{actor_id}/builds/default'


def _prune_actor_input_schema(
    input_schema: dict,
    max_description_len: int = _MAX_DESCRIPTION_LEN,
) -> tuple[dict, list[str]]:
    """Get the input schema from the Actor build.

    Trim descriptions to ``_MAX_DESCRIPTION_LEN`` characters.

    Args:
        input_schema (dict): The input schema from the Actor build.
        max_description_len (int): The maximum length of the description.

    Returns:
        tuple[dict, list[str]]: A tuple containing the pruned properties
            and required fields.
    """
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])

    properties_out: dict = {}
    for item, meta in properties.items():
        properties_out[item] = {}
        if desc := meta.get('description'):
            properties_out[item]['description'] = (
                desc[:max_description_len] + '...' if len(desc) > max_description_len else desc
            )
        for key_name in ('type', 'default', 'prefill', 'enum'):
            if value := meta.get(key_name):
                properties_out[item][key_name] = value

    return properties_out, required


T = TypeVar('T', ApifyClient, ApifyClientAsync)


def _create_apify_client(client_cls: type[T], token: str) -> T:
    """Create an Apify client instance with a custom user-agent.

    Args:
        client_cls (ApifyClient | ApifyClientAsync): ApifyClient or ApifyClientAsync class.
        token (str): API token.

    Returns:
        T: ApifyClient or ApifyClientAsync instance.

    Raises:
        ValueError: If the API token is not provided.
    """
    if not token:
        msg = 'API token is required to create an Apify client.'
        raise ValueError(msg)
    client = client_cls(token)

    # Check for new attribute names first (without 'x'), then fall back to old names (with 'x')
    if isinstance(client, ApifyClientAsync):
        http_client_attr = (
            'httpx_async_client' if hasattr(client.http_client, 'httpx_async_client') else 'http_async_client'
        )
    else:
        http_client_attr = 'httpx_client' if hasattr(client.http_client, 'httpx_client') else 'http_client'

    if http_client := getattr(client.http_client, http_client_attr, None):
        http_client.headers['user-agent'] += '; Origin/langchain'
    return client


def _actor_id_to_tool_name(actor_id: str) -> str:
    """Turn actor_id into a valid tool name.

    Tool name must only contain letters, numbers, underscores, dashes,
    and cannot contain spaces.

    Args:
        actor_id (str): Actor ID from Apify store.

    Returns:
        str: A valid tool name.
    """
    valid_chars = string.ascii_letters + string.digits + '_-'
    return 'apify_actor_' + ''.join(char if char in valid_chars else '_' for char in actor_id)


def _get_actor_latest_build(apify_client: ApifyClient, actor_id: str) -> dict:
    """Get the latest build of an Actor from the default build tag.

    Args:
        apify_client (ApifyClient): An instance of the ApifyClient class.
        actor_id (str): Actor name from Apify store to run.

    Returns:
        dict: The latest build of the Actor.

    Raises:
        ValueError: If the Actor is not found or the build data is not found.
        TypeError: If the build is not a dictionary.
    """
    if not (actor := apify_client.actor(actor_id).get()):
        msg = f'Actor {actor_id} not found.'
        raise ValueError(msg)

    if not (actor_obj_id := actor.get('id')):
        msg = f'Failed to get the Actor object ID for {actor_id}.'
        raise ValueError(msg)

    url = _APIFY_API_ENDPOINT_GET_DEFAULT_BUILD.format(actor_id=actor_obj_id)
    response = requests.request('GET', url, timeout=_REQUESTS_TIMEOUT_SECS)

    build = response.json()
    if not isinstance(build, dict):
        msg = f'Failed to get the latest build of the Actor {actor_id}.'
        raise TypeError(msg)

    if (data := build.get('data')) is None:
        msg = f'Failed to get the latest build data of the Actor {actor_id}.'
        raise ValueError(msg)

    return data


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
