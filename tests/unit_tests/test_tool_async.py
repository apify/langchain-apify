"""Under `ainvoke`, the tools run on the async Apify client instead of blocking a thread.

Each generic tool awaits the `a`-prefixed version of the `ApifyToolsClient` method it calls under `invoke`, with the
same arguments, and returns the same output.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apify_client import ApifyClientAsync

from langchain_apify import (
    ApifyActorsTool,
    ApifyEcommerceScraperTool,
    ApifyFacebookPostsScraperTool,
    ApifyGetDatasetItemsTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyInstagramScraperTool,
    ApifyLinkedInProfileDetailTool,
    ApifyLinkedInProfilePostsTool,
    ApifyLinkedInProfileSearchTool,
    ApifyRAGWebBrowserTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetDatasetTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
    ApifyTikTokScraperTool,
    ApifyTwitterScraperTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
)
from langchain_apify._client import ApifyToolsClient
from tests.unit_tests.conftest import SAMPLE_ITEMS, SUCCEEDED_RUN, make_items_page, make_tool

if TYPE_CHECKING:
    from collections.abc import Iterator

_TOOL_CASES = [
    pytest.param(ApifyRunActorTool, 'run_actor', {'actor_id': 'apify/test'}, id='run_actor'),
    pytest.param(ApifyGetDatasetItemsTool, 'get_dataset_items', {'dataset_id': 'dataset-xyz'}, id='dataset_items'),
    pytest.param(
        ApifyRunActorAndGetDatasetTool, 'run_actor_and_get_items', {'actor_id': 'apify/test'}, id='run_actor_and_items'
    ),
    pytest.param(ApifyScrapeUrlTool, 'scrape_url_with_metadata', {'url': 'https://example.com'}, id='scrape_url'),
    pytest.param(ApifyRunTaskTool, 'run_task', {'task_id': 'user/my-task'}, id='run_task'),
    pytest.param(
        ApifyRunTaskAndGetDatasetTool, 'run_task_and_get_items', {'task_id': 'user/my-task'}, id='run_task_and_items'
    ),
    pytest.param(ApifyGoogleSearchTool, 'google_search', {'query': 'apify'}, id='google_search'),
    pytest.param(ApifyWebCrawlerTool, 'crawl_website', {'url': 'https://example.com'}, id='web_crawler'),
    pytest.param(ApifyRAGWebBrowserTool, 'rag_web_search', {'query': 'langchain'}, id='rag_web_browser'),
    pytest.param(ApifyGoogleMapsTool, 'google_maps_search', {'query': 'coffee'}, id='google_maps'),
    pytest.param(ApifyYouTubeScraperTool, 'youtube_scrape', {'search_query': 'langchain'}, id='youtube'),
    pytest.param(ApifyEcommerceScraperTool, 'ecommerce_scrape', {'url': 'https://shop.example.com/p/1'}, id='shop'),
    pytest.param(
        ApifyInstagramScraperTool,
        'instagram_scrape',
        {'search_type': 'user', 'search_query': 'apify'},
        id='instagram',
    ),
    pytest.param(
        ApifyLinkedInProfilePostsTool, 'linkedin_profile_posts', {'profile_url': 'apify'}, id='linkedin_posts'
    ),
    pytest.param(ApifyLinkedInProfileSearchTool, 'linkedin_profile_search', {'query': 'apify'}, id='linkedin_search'),
    pytest.param(
        ApifyLinkedInProfileDetailTool, 'linkedin_profile_detail', {'profile_url': 'apify'}, id='linkedin_detail'
    ),
    pytest.param(ApifyTwitterScraperTool, 'twitter_scrape', {'search_query': 'apify'}, id='twitter'),
    pytest.param(ApifyTikTokScraperTool, 'tiktok_scrape', {'search_query': 'apify'}, id='tiktok'),
    pytest.param(
        ApifyFacebookPostsScraperTool, 'facebook_posts_scrape', {'page_url': 'https://facebook.com/apify'}, id='fb'
    ),
]


def _client_result(method: str) -> object:
    """What the client method returns, in the shape the tool expects."""
    if method in {'run_actor', 'run_task'}:
        return SUCCEEDED_RUN
    if method == 'get_dataset_items':
        return SAMPLE_ITEMS
    if method == 'scrape_url_with_metadata':
        return (SUCCEEDED_RUN, [{'markdown': '# content'}], '# content', 'markdown')
    return (SUCCEEDED_RUN, SAMPLE_ITEMS)


@pytest.mark.parametrize(('tool_cls', 'method', 'tool_input'), _TOOL_CASES)
async def test_ainvoke_matches_invoke_on_the_async_client(
    mock_tools_client: MagicMock, tool_cls: type, method: str, tool_input: dict
) -> None:
    """`ainvoke` awaits the async client method with `invoke`'s arguments, never the sync one, and returns the same."""
    async_method = f'a{method}'
    assert inspect.iscoroutinefunction(getattr(ApifyToolsClient, async_method, None)), async_method
    getattr(mock_tools_client, method).return_value = _client_result(method)
    setattr(mock_tools_client, async_method, AsyncMock(return_value=_client_result(method)))
    tool = make_tool(tool_cls, mock_tools_client)

    async_output = await tool.ainvoke(tool_input)
    getattr(mock_tools_client, method).assert_not_called()
    sync_output = tool.invoke(tool_input)

    assert async_output == sync_output
    sync_call = getattr(mock_tools_client, method).call_args
    getattr(mock_tools_client, async_method).assert_awaited_once_with(*sync_call.args, **sync_call.kwargs)


async def test_ainvoke_reports_errors_like_invoke(mock_tools_client: MagicMock) -> None:
    """A failed run reaches the agent the same way on both paths."""
    mock_tools_client.run_actor.side_effect = RuntimeError('run failed')
    mock_tools_client.arun_actor = AsyncMock(side_effect=RuntimeError('run failed'))
    tool = make_tool(ApifyRunActorTool, mock_tools_client)

    assert await tool.ainvoke({'actor_id': 'apify/test'}) == tool.invoke({'actor_id': 'apify/test'})


@pytest.fixture
def actors_tool_clients() -> Iterator[tuple[MagicMock, MagicMock]]:
    """Mocks of the sync and async Apify clients that `ApifyActorsTool` creates."""
    sync_client, async_client = MagicMock(), MagicMock()
    async_client.actor.return_value.call = AsyncMock(return_value=SUCCEEDED_RUN)
    list_items = AsyncMock(return_value=make_items_page(SAMPLE_ITEMS))
    async_client.run.return_value.dataset.return_value.list_items = list_items
    build = {'actorDefinition': {'input': {'type': 'object', 'properties': {}}}}
    with (
        patch(
            'langchain_apify.tools.actors._create_apify_client',
            side_effect=lambda cls, _token: async_client if cls is ApifyClientAsync else sync_client,
        ),
        patch('langchain_apify.tools.actors._get_actor_latest_build', return_value=build),
    ):
        yield sync_client, async_client


async def test_actors_tool_ainvoke_runs_on_the_async_client(actors_tool_clients: tuple[MagicMock, MagicMock]) -> None:
    """`ApifyActorsTool` runs the Actor on the async client, with its charge limit, and returns the dataset items."""
    sync_client, async_client = actors_tool_clients
    tool = ApifyActorsTool('apify/hello-world', apify_token='dummy-token', max_total_charge_usd=0.25)

    items = await tool.ainvoke({'run_input': '{"a": 1}'})

    assert items == SAMPLE_ITEMS
    sync_client.actor.return_value.call.assert_not_called()
    call_kwargs = async_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs['run_input'] == {'a': 1}
    assert str(call_kwargs['max_total_charge_usd']) == '0.25'
    async_client.run.assert_called_once_with(run_id=SUCCEEDED_RUN.id)
