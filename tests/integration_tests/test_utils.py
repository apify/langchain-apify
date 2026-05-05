import os

from apify_client.client import ApifyClient

from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import _create_apify_client, _get_actor_latest_build


def test_get_actor_latest_build() -> None:
    """Tests the get_actor_latest_build function.

    Raises:
        ValueError: If the APIFY_API_TOKEN environment variable is not set.
    """
    if (token := os.getenv('APIFY_API_TOKEN')) is None:
        msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
        raise ValueError(msg)

    apify_client = _create_apify_client(ApifyClient, token)

    build = _get_actor_latest_build(apify_client, 'apify/rag-web-browser')

    assert isinstance(build, dict)
    assert 'id' in build
    assert 'description' in build.get('actorDefinition', {})
