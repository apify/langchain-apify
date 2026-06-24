from __future__ import annotations

import os
import string
import warnings
from typing import TypeVar

import requests
from apify_client import ApifyClientAsync
from apify_client.client import ApifyClient
from pydantic import SecretStr

_MAX_DESCRIPTION_LEN: int = 350

_DEPRECATED_APIFY_API_TOKEN_MSG = "The 'apify_api_token' parameter is deprecated, use 'apify_token' instead."
_DEPRECATED_APIFY_API_TOKEN_ENV_MSG = (
    "The 'APIFY_API_TOKEN' environment variable is deprecated, use 'APIFY_TOKEN' instead."
)
_BOTH_TOKENS_MSG = (
    "Both 'apify_token' and 'apify_api_token' were specified; using 'apify_token' "
    "and ignoring the deprecated 'apify_api_token'."
)
_REQUESTS_TIMEOUT_SECS: float = 10.0
_APIFY_API_ENDPOINT_GET_DEFAULT_BUILD: str = 'https://api.apify.com/v2/acts/{actor_id}/builds/default'


def _resolve_deprecated_token(
    apify_token: SecretStr | str | None,
    apify_api_token: SecretStr | str | None,
) -> SecretStr | str | None:
    """Apply the ``apify_api_token`` → ``apify_token`` deprecation policy.

    For classes with an explicit ``__init__`` (``ApifyToolsClient``,
    ``ApifyDatasetLoader``, ``ApifyCrawlLoader``, ``ApifyActorsTool``). Emits a
    ``DeprecationWarning`` when the legacy ``apify_api_token`` is supplied and
    prefers ``apify_token`` when both are given. Returns the token to use.
    """
    if apify_api_token is None:
        return apify_token
    if apify_token is not None:
        warnings.warn(_BOTH_TOKENS_MSG, DeprecationWarning, stacklevel=3)
        return apify_token
    warnings.warn(_DEPRECATED_APIFY_API_TOKEN_MSG, DeprecationWarning, stacklevel=3)
    return apify_api_token


def _resolve_deprecated_token_values(values: dict) -> dict:
    """Same deprecation policy as :func:`_resolve_deprecated_token`, for dicts.

    For pydantic ``model_validator(mode='before')`` hooks (``_ApifyGenericTool``,
    ``ApifySearchRetriever``), which receive the raw input ``values`` dict.
    """
    if isinstance(values, dict) and 'apify_api_token' in values:
        if 'apify_token' in values:
            warnings.warn(_BOTH_TOKENS_MSG, DeprecationWarning, stacklevel=3)
            del values['apify_api_token']
        else:
            warnings.warn(_DEPRECATED_APIFY_API_TOKEN_MSG, DeprecationWarning, stacklevel=3)
            values['apify_token'] = values.pop('apify_api_token')
    return values


def _resolve_apify_token() -> str | None:
    """Resolve the Apify API token from environment variables.

    ``APIFY_TOKEN`` (SDK-standard) takes precedence; ``APIFY_API_TOKEN`` is
    kept as a deprecated fallback for backwards compatibility with this
    package's historical naming, and emits a ``DeprecationWarning`` when used.
    """
    if token := os.getenv('APIFY_TOKEN'):
        return token
    if token := os.getenv('APIFY_API_TOKEN'):
        warnings.warn(_DEPRECATED_APIFY_API_TOKEN_ENV_MSG, DeprecationWarning, stacklevel=2)
        return token
    return None


def _apify_token_secret_factory() -> SecretStr | None:
    """Pydantic ``default_factory`` returning the resolved token as ``SecretStr``."""
    token = _resolve_apify_token()
    return SecretStr(token) if token else None


def _extract_content(item: dict) -> str:
    """Return an Actor item's content, preferring markdown over plain text.

    Both ``apify/website-content-crawler`` and ``apify/rag-web-browser`` emit
    ``markdown`` (the richer field) and ``text`` (the plain-text fallback). The
    trailing ``or ''`` guarantees a string even when a key is present but null.
    """
    return item.get('markdown') or item.get('text') or ''


def _item_metadata(item: dict) -> dict:
    """Return an item's ``metadata`` block, or ``{}`` if missing/non-dict.

    Some Actors surface a ``null`` (or otherwise non-dict) ``metadata`` value,
    so a plain ``item.get('metadata', {})`` would raise ``AttributeError`` on
    the chained ``.get(...)``.
    """
    meta = item.get('metadata')
    return meta if isinstance(meta, dict) else {}


def _safe_title(item: dict) -> str:
    """Return an Actor item's title from its nested ``metadata`` object.

    Both ``apify/website-content-crawler`` and ``apify/rag-web-browser`` nest
    the page title under ``metadata.title``. The guard tolerates Actor
    responses where ``metadata`` is missing or not a dict.
    """
    return _item_metadata(item).get('title', '')


def _extract_source(item: dict) -> str:
    """Return an Actor item's source URL via one canonical fallback order.

    ``apify/rag-web-browser`` items expose the page URL in several places. To
    keep every consumer (RAG tool, retriever, loaders) in agreement, the order
    is fixed here: nested ``metadata.url`` first, then ``crawledUrl``, then the
    top-level ``url``.
    """
    return _item_metadata(item).get('url') or item.get('crawledUrl') or item.get('url', '')


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
            if (value := meta.get(key_name)) is not None:
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
