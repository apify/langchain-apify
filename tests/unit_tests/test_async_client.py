"""The async side of `ApifyToolsClient` runs Actors on `ApifyClientAsync` instead of blocking a thread.

Each domain method builds its Actor call once (`*_call`), and `run_call` and `arun_call` run it, so the sync and async
paths send the same input. The async Apify client is created on the first async call only.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import impit
import pytest
from apify_client import ApifyClient, ApifyClientAsync

from langchain_apify._client import ApifyToolsClient
from tests.unit_tests.conftest import FAILED_RUN, SAMPLE_ITEMS, SUCCEEDED_RUN, make_items_page

if TYPE_CHECKING:
    from collections.abc import Iterator


def _async_apify_client() -> MagicMock:
    client = MagicMock()
    client.actor.return_value.call = AsyncMock(return_value=SUCCEEDED_RUN)
    client.task.return_value.call = AsyncMock(return_value=SUCCEEDED_RUN)
    client.dataset.return_value.list_items = AsyncMock(return_value=make_items_page(SAMPLE_ITEMS))
    return client


@pytest.fixture
def apify_clients() -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    """Mocks of the sync and async Apify clients, and of `_create_apify_client` that hands them out."""
    sync_client, async_client = MagicMock(), _async_apify_client()
    sync_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    sync_client.dataset.return_value.list_items.return_value = make_items_page(SAMPLE_ITEMS)
    create = MagicMock(side_effect=lambda cls, _token: async_client if cls is ApifyClientAsync else sync_client)
    with patch('langchain_apify._client._create_apify_client', create):
        yield sync_client, async_client, create


Clients = tuple[MagicMock, MagicMock, MagicMock]


async def test_async_client_is_created_once_on_first_async_call(apify_clients: Clients) -> None:
    """Sync-only use never creates the async client; async calls share one."""
    _, _, create = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    assert [call.args[0] for call in create.call_args_list] == [ApifyClient]

    await client.arun_actor('apify/hello-world')
    await client.arun_actor('apify/hello-world')

    assert [call.args[0] for call in create.call_args_list] == [ApifyClient, ApifyClientAsync]


async def test_arun_actor_sends_what_run_actor_sends(apify_clients: Clients) -> None:
    """The async run carries the same input, timeout, memory and charge limit as the sync one."""
    sync_client, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token', max_total_charge_usd=0.1)

    client.run_actor('apify/hello-world', {'a': 1}, timeout_secs=60, memory_mbytes=512)
    run = await client.arun_actor('apify/hello-world', {'a': 1}, timeout_secs=60, memory_mbytes=512)

    assert run == SUCCEEDED_RUN
    async_kwargs = async_client.actor.return_value.call.call_args.kwargs
    assert async_kwargs == sync_client.actor.return_value.call.call_args.kwargs
    assert async_kwargs == {
        'run_input': {'a': 1},
        'run_timeout': timedelta(seconds=60),
        'logger': None,
        'memory_mbytes': 512,
        'max_total_charge_usd': Decimal('0.1'),
    }


async def test_arun_actor_raises_when_the_run_fails(apify_clients: Clients) -> None:
    """A run that doesn't end with `SUCCEEDED` raises, as in the sync path."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    async_client.actor.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        await client.arun_actor('apify/hello-world')


async def test_arun_actor_wraps_transport_errors(apify_clients: Clients) -> None:
    """Network errors become `RuntimeError`, as in the sync path."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    async_client.actor.return_value.call.side_effect = impit.ConnectError('conn refused')

    with pytest.raises(RuntimeError, match='Apify Actor call failed'):
        await client.arun_actor('apify/hello-world')


async def test_arun_actor_raises_without_run_details(apify_clients: Clients) -> None:
    """A call that returns no run raises, as in the sync path."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    async_client.actor.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match='returned no run details'):
        await client.arun_actor('apify/hello-world')


async def test_aget_dataset_items_lists_clean_items(apify_clients: Clients) -> None:
    """Dataset items come from the async client with the same arguments as the sync path."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')

    items = await client.aget_dataset_items('dataset-xyz', limit=50, offset=10)

    assert items == SAMPLE_ITEMS
    async_client.dataset.return_value.list_items.assert_awaited_once_with(limit=50, offset=10, clean=True)


async def test_arun_actor_and_get_items_reads_the_run_dataset(apify_clients: Clients) -> None:
    """The items come from the run's default dataset."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')

    run, items = await client.arun_actor_and_get_items('apify/hello-world', dataset_items_limit=5)

    assert (run, items) == (SUCCEEDED_RUN, SAMPLE_ITEMS)
    async_client.dataset.assert_called_once_with('dataset-xyz')


async def test_arun_task_and_get_items_runs_the_task(apify_clients: Clients) -> None:
    """Task runs go through the async client with the task input and the charge limit."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token', max_total_charge_usd=0.1)

    run, items = await client.arun_task_and_get_items('user/my-task', {'a': 1})

    assert (run, items) == (SUCCEEDED_RUN, SAMPLE_ITEMS)
    kwargs = async_client.task.return_value.call.call_args.kwargs
    assert kwargs['task_input'] == {'a': 1}
    assert kwargs['max_total_charge_usd'] == Decimal('0.1')


async def test_arun_call_matches_the_sync_domain_method(apify_clients: Clients) -> None:
    """A domain call built once returns the same post-processed results on both paths."""
    sync_client, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    google_items = [{'organicResults': [{'title': 'Apify', 'url': 'https://apify.com', 'description': 'd'}]}]
    sync_client.dataset.return_value.list_items.return_value = make_items_page(google_items)
    async_client.dataset.return_value.list_items.return_value = make_items_page(google_items)

    sync_result = client.google_search('apify', max_results=1)
    async_result = await client.arun_call(client.google_search_call('apify', max_results=1))

    assert async_result == sync_result
    assert async_result[1] == [{'title': 'Apify', 'url': 'https://apify.com', 'description': 'd'}]
    assert (
        async_client.actor.return_value.call.call_args.kwargs['run_input']
        == sync_client.actor.return_value.call.call_args.kwargs['run_input']
    )


async def test_ascrape_url_with_metadata_returns_the_page_content(apify_clients: Clients) -> None:
    """The async scrape returns the page content and where it came from, like the sync one."""
    _, async_client, _ = apify_clients
    client = ApifyToolsClient(apify_token='dummy-token')
    async_client.dataset.return_value.list_items.return_value = make_items_page([{'markdown': '# Apify'}])

    run, items, content, source = await client.ascrape_url_with_metadata('https://apify.com')

    assert run == SUCCEEDED_RUN
    assert items == [{'markdown': '# Apify'}]
    assert (content, source) == ('# Apify', 'markdown')
