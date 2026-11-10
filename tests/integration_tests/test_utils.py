import pytest
from apify_client import ApifyClient

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import _create_apify_client, _get_actor_latest_build, _resolve_apify_token


def test_get_actor_latest_build() -> None:
    """Tests the get_actor_latest_build function.

    Raises:
        ValueError: If the APIFY_TOKEN environment variable is not set.
    """
    if (token := _resolve_apify_token()) is None:
        # ty can't read pytest 7's `skip` signature, which is typed through a Protocol attribute.
        pytest.skip(_ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET)  # ty: ignore[too-many-positional-arguments]

    apify_client = _create_apify_client(ApifyClient, token)

    build = _get_actor_latest_build(apify_client, 'apify/rag-web-browser')

    assert isinstance(build, dict)
    assert 'id' in build
    assert 'description' in build.get('actorDefinition', {})
