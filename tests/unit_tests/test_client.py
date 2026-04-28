from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from langchain_apify._client import ApifyToolsClient
from tests.unit_tests.conftest import FAILED_RUN, SAMPLE_ITEMS, SUCCEEDED_RUN

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_with_explicit_token(mock_apify_client: MagicMock) -> None:
    with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client) as mock_create:
        c = ApifyToolsClient(apify_api_token='my-token')
        mock_create.assert_called_once()
        assert c._client is mock_apify_client


def test_init_empty_token_raises() -> None:
    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        ApifyToolsClient(apify_api_token='')


# ---------------------------------------------------------------------------
# run_actor
# ---------------------------------------------------------------------------


def test_run_actor_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN

    result = client.run_actor('apify/test-actor', run_input={'key': 'val'})

    mock_apify_client.actor.assert_called_once_with('apify/test-actor')
    mock_apify_client.actor.return_value.call.assert_called_once_with(
        run_input={'key': 'val'}, timeout_secs=300, logger=None
    )
    assert result == SUCCEEDED_RUN


def test_run_actor_with_memory(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN

    client.run_actor('apify/test-actor', memory_mbytes=512)

    mock_apify_client.actor.return_value.call.assert_called_once_with(
        run_input=None, timeout_secs=300, logger=None, memory_mbytes=512
    )


def test_run_actor_failed_status_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        client.run_actor('apify/test-actor')


# ---------------------------------------------------------------------------
# get_dataset_items
# ---------------------------------------------------------------------------


def test_get_dataset_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    items = client.get_dataset_items('dataset-xyz', limit=50, offset=10)

    mock_apify_client.dataset.assert_called_once_with('dataset-xyz')
    mock_apify_client.dataset.return_value.list_items.assert_called_once_with(limit=50, offset=10, clean=True)
    assert items == SAMPLE_ITEMS


def test_get_dataset_items_empty(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.return_value.items = []

    items = client.get_dataset_items('dataset-empty')
    assert items == []


# ---------------------------------------------------------------------------
# run_actor_and_get_items
# ---------------------------------------------------------------------------


def test_run_actor_and_get_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    run, items = client.run_actor_and_get_items('apify/test-actor', run_input={'q': '1'})

    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS
    mock_apify_client.dataset.assert_called_once_with('dataset-xyz')


def test_run_actor_and_get_items_missing_dataset_id_raises(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    run_no_dataset = {**SUCCEEDED_RUN, 'defaultDatasetId': None}
    mock_apify_client.actor.return_value.call.return_value = run_no_dataset

    with pytest.raises(RuntimeError, match='no default dataset ID'):
        client.run_actor_and_get_items('apify/test-actor')


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------


def test_run_task_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN

    result = client.run_task('user/my-task', task_input={'key': 'val'})

    mock_apify_client.task.assert_called_once_with('user/my-task')
    mock_apify_client.task.return_value.call.assert_called_once_with(task_input={'key': 'val'}, timeout_secs=300)
    assert result == SUCCEEDED_RUN


def test_run_task_failed_status_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = FAILED_RUN

    with pytest.raises(RuntimeError, match='run-fail'):
        client.run_task('user/my-task')


# ---------------------------------------------------------------------------
# run_task_and_get_items
# ---------------------------------------------------------------------------


def test_run_task_and_get_items_success(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = SAMPLE_ITEMS

    run, items = client.run_task_and_get_items('user/my-task')

    assert run == SUCCEEDED_RUN
    assert items == SAMPLE_ITEMS


def test_run_task_and_get_items_missing_dataset_id_raises(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    run_no_dataset = {**SUCCEEDED_RUN, 'defaultDatasetId': None}
    mock_apify_client.task.return_value.call.return_value = run_no_dataset

    with pytest.raises(RuntimeError, match='no default dataset ID'):
        client.run_task_and_get_items('user/my-task')


# ---------------------------------------------------------------------------
# scrape_url
# ---------------------------------------------------------------------------


def test_scrape_url_returns_markdown(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'markdown': '# Hello', 'text': 'Hello', 'url': 'https://example.com'},
    ]

    content = client.scrape_url('https://example.com')
    assert content == '# Hello'


def test_scrape_url_falls_back_to_text(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'text': 'Plain text content', 'url': 'https://example.com'},
    ]

    content = client.scrape_url('https://example.com')
    assert content == 'Plain text content'


def test_scrape_url_empty_items_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = []

    with pytest.raises(RuntimeError, match='No content extracted'):
        client.scrape_url('https://example.com')


def test_scrape_url_empty_content_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.return_value.items = [
        {'markdown': '', 'text': '', 'url': 'https://example.com'},
    ]

    with pytest.raises(RuntimeError, match='No content extracted'):
        client.scrape_url('https://example.com')


# ---------------------------------------------------------------------------
# _check_run_status
# ---------------------------------------------------------------------------


def test_check_run_status_succeeded() -> None:
    ApifyToolsClient._check_run_status({'id': 'run-ok', 'status': 'SUCCEEDED'})


def test_check_run_status_failed() -> None:
    with pytest.raises(RuntimeError, match='run-bad'):
        ApifyToolsClient._check_run_status({'id': 'run-bad', 'status': 'FAILED'})


# ---------------------------------------------------------------------------
# None returns from actor/task .call()
# ---------------------------------------------------------------------------


def test_run_actor_none_return_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match='returned no run details'):
        client.run_actor('apify/broken-actor')


def test_run_task_none_return_raises(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match='returned no run details'):
        client.run_task('user/broken-task')


# ---------------------------------------------------------------------------
# Transport-error wrapping (httpx / ApifyClientError -> RuntimeError)
# ---------------------------------------------------------------------------


def test_run_actor_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.actor.return_value.call.side_effect = httpx.ConnectError('conn refused')

    with pytest.raises(RuntimeError, match='Apify Actor call failed'):
        client.run_actor('apify/test-actor')


def test_get_dataset_items_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('timeout')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.get_dataset_items('dataset-xyz')


def test_run_actor_and_get_items_dataset_fetch_network_error(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    mock_apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('reset')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.run_actor_and_get_items('apify/test-actor')


def test_run_task_network_error_wraps(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    mock_apify_client.task.return_value.call.side_effect = httpx.ConnectError('conn refused')

    with pytest.raises(RuntimeError, match='Apify task call failed'):
        client.run_task('user/my-task')


def test_run_task_and_get_items_dataset_fetch_network_error(
    client: ApifyToolsClient, mock_apify_client: MagicMock
) -> None:
    mock_apify_client.task.return_value.call.return_value = SUCCEEDED_RUN
    mock_apify_client.dataset.return_value.list_items.side_effect = httpx.ConnectError('reset')

    with pytest.raises(RuntimeError, match='Apify dataset fetch failed'):
        client.run_task_and_get_items('user/my-task')


def test_run_actor_programming_error_propagates(client: ApifyToolsClient, mock_apify_client: MagicMock) -> None:
    """Non-transport exceptions (programming errors) must NOT be wrapped as RuntimeError."""
    mock_apify_client.actor.return_value.call.side_effect = AttributeError('bug in SDK')

    with pytest.raises(AttributeError, match='bug in SDK'):
        client.run_actor('apify/test-actor')
