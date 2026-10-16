"""``ApifyWrapper`` has to hand apify-client the right timeout argument.

apify-client 3.x takes the run timeout as a ``timedelta`` under ``run_timeout``, and has a separate ``timeout``
argument that bounds the HTTP request instead of the run. Passing seconds, or passing the value under the wrong
name, fails silently: the call still succeeds and the run just is not bounded the way the caller asked.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_apify import ApifyWrapper
from langchain_apify._utils import _run_timeout
from tests.unit_tests.conftest import SUCCEEDED_RUN

if TYPE_CHECKING:
    from collections.abc import Iterator

_TIMEOUT_CASES = [(300, timedelta(seconds=300)), (None, None)]


def _noop_mapping(item: dict) -> Document:
    return Document(page_content=item.get('text', ''))


@pytest.fixture
def wrapper() -> Iterator[tuple[ApifyWrapper, MagicMock, MagicMock]]:
    """An ``ApifyWrapper`` whose sync and async Apify clients are mocks."""
    sync_client, async_client = MagicMock(), MagicMock()
    sync_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    sync_client.task.return_value.call.return_value = SUCCEEDED_RUN
    async_client.actor.return_value.call = AsyncMock(return_value=SUCCEEDED_RUN)
    async_client.task.return_value.call = AsyncMock(return_value=SUCCEEDED_RUN)

    # The validator creates the sync client first, then the async one.
    with patch('langchain_apify.wrappers._create_apify_client', side_effect=[sync_client, async_client]):
        yield ApifyWrapper(apify_token='test-token'), sync_client, async_client


def _assert_run_timeout(call_mock: MagicMock, expected: timedelta | None) -> None:
    kwargs = call_mock.call_args.kwargs
    assert kwargs['run_timeout'] == expected
    # `timeout` would cap the HTTP request instead of the run.
    assert 'timeout' not in kwargs


@pytest.mark.parametrize(('timeout_secs', 'expected'), _TIMEOUT_CASES)
def test_call_actor_passes_run_timeout(
    wrapper: tuple[ApifyWrapper, MagicMock, MagicMock],
    timeout_secs: int | None,
    expected: timedelta | None,
) -> None:
    apify_wrapper, sync_client, _ = wrapper
    apify_wrapper.call_actor(
        actor_id='apify/hello-world',
        run_input={},
        dataset_mapping_function=_noop_mapping,
        timeout_secs=timeout_secs,
    )
    _assert_run_timeout(sync_client.actor.return_value.call, expected)


@pytest.mark.parametrize(('timeout_secs', 'expected'), _TIMEOUT_CASES)
async def test_acall_actor_passes_run_timeout(
    wrapper: tuple[ApifyWrapper, MagicMock, MagicMock],
    timeout_secs: int | None,
    expected: timedelta | None,
) -> None:
    apify_wrapper, _, async_client = wrapper
    await apify_wrapper.acall_actor(
        actor_id='apify/hello-world',
        run_input={},
        dataset_mapping_function=_noop_mapping,
        timeout_secs=timeout_secs,
    )
    _assert_run_timeout(async_client.actor.return_value.call, expected)


@pytest.mark.parametrize(('timeout_secs', 'expected'), _TIMEOUT_CASES)
def test_call_actor_task_passes_run_timeout(
    wrapper: tuple[ApifyWrapper, MagicMock, MagicMock],
    timeout_secs: int | None,
    expected: timedelta | None,
) -> None:
    apify_wrapper, sync_client, _ = wrapper
    apify_wrapper.call_actor_task(
        task_id='user/my-task',
        task_input={},
        dataset_mapping_function=_noop_mapping,
        timeout_secs=timeout_secs,
    )
    _assert_run_timeout(sync_client.task.return_value.call, expected)


@pytest.mark.parametrize(('timeout_secs', 'expected'), _TIMEOUT_CASES)
async def test_acall_actor_task_passes_run_timeout(
    wrapper: tuple[ApifyWrapper, MagicMock, MagicMock],
    timeout_secs: int | None,
    expected: timedelta | None,
) -> None:
    apify_wrapper, _, async_client = wrapper
    await apify_wrapper.acall_actor_task(
        task_id='user/my-task',
        task_input={},
        dataset_mapping_function=_noop_mapping,
        timeout_secs=timeout_secs,
    )
    _assert_run_timeout(async_client.task.return_value.call, expected)


@pytest.mark.parametrize(('timeout_secs', 'expected'), _TIMEOUT_CASES)
def test_run_timeout_converts_seconds(timeout_secs: int | None, expected: timedelta | None) -> None:
    assert _run_timeout(timeout_secs) == expected
