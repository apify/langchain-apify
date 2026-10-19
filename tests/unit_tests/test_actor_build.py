"""``_get_actor_latest_build`` has to keep returning the raw API payload shape.

The build now comes from `apify_client.actor(...).default_build().get()` as a pydantic `Build` rather than from a
hand-built HTTP request, but `ApifyActorsTool` still reads it as the dict the Apify API returns. apify-client does
not model every field of a build, so these tests pin the two keys the tool depends on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from apify_client._models import Build
from apify_client.errors import NotFoundError

from langchain_apify._utils import _get_actor_latest_build
from langchain_apify.tools import ApifyActorsTool

_ACTOR_ID = 'apify/hello-world'

_BUILD_PAYLOAD: dict = {
    'id': 'build-abc',
    'actId': 'actor-abc',
    'userId': 'user-abc',
    'status': 'SUCCEEDED',
    'startedAt': '2025-01-01T00:00:00.000Z',
    'finishedAt': '2025-01-01T00:01:00.000Z',
    'buildNumber': '0.1.2',
    'meta': {'origin': 'API'},
    'actorDefinition': {
        'actorSpecification': 1,
        'name': 'hello-world',
        'version': '0.1',
        # `ActorDefinition` has no `description` field, so this one only survives as a pydantic extra.
        'description': 'An Actor that greets.',
        'input': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
    },
}


def _client_returning(build: Build | None) -> MagicMock:
    client = MagicMock()
    client.actor.return_value.default_build.return_value.get.return_value = build
    return client


def test_build_is_fetched_through_the_client() -> None:
    client = _client_returning(Build.model_validate(_BUILD_PAYLOAD))

    _get_actor_latest_build(client, _ACTOR_ID)

    client.actor.assert_called_once_with(_ACTOR_ID)
    client.actor.return_value.default_build.assert_called_once_with()


def test_build_keeps_the_api_key_names() -> None:
    result = _get_actor_latest_build(_client_returning(Build.model_validate(_BUILD_PAYLOAD)), _ACTOR_ID)

    assert result['id'] == 'build-abc'
    assert result['actorDefinition']['description'] == 'An Actor that greets.'
    assert result['actorDefinition']['input']['required'] == ['url']


def test_build_feeds_the_actor_tool_readers() -> None:
    """The dict has to satisfy both `ApifyActorsTool` readers; that is the whole contract of this function."""
    result = _get_actor_latest_build(_client_returning(Build.model_validate(_BUILD_PAYLOAD)), _ACTOR_ID)

    assert ApifyActorsTool._create_description(result) == 'An Actor that greets.'
    args_schema = ApifyActorsTool._build_tool_args_schema_model(result, _ACTOR_ID)
    assert 'run_input' in args_schema.model_fields


def test_missing_build_raises_value_error() -> None:
    with pytest.raises(ValueError, match='Failed to get the latest build data'):
        _get_actor_latest_build(_client_returning(None), _ACTOR_ID)


def test_missing_actor_raises_value_error() -> None:
    """A missing Actor reaches us as apify-client's `NotFoundError`; callers expect a `ValueError`."""
    response = MagicMock(status_code=404, text='{}')
    response.json.return_value = {'error': {'message': 'Actor with this name was not found'}}
    client = _client_returning(None)
    client.actor.return_value.default_build.return_value.get.side_effect = NotFoundError(response, attempt=1)

    with pytest.raises(ValueError, match=f'Actor {_ACTOR_ID} not found'):
        _get_actor_latest_build(client, _ACTOR_ID)
