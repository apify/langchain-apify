import os

from apify_client.client import ApifyClient

from langchain_apify.error_messages import ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify.utils import create_apify_client, get_actor_latest_build


def test_get_actor_latest_build() -> None:
    """Tests the get_actor_latest_build function."""
    if (token := os.getenv('APIFY_API_TOKEN')) is None:
        msg = ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
        raise ValueError(msg)

    apify_client = create_apify_client(ApifyClient, token)

    build = get_actor_latest_build(apify_client, 'apify/rag-web-browser')

    assert isinstance(build, dict)
    assert 'id' in build
    assert 'description' in build.get('actorDefinition', {})
