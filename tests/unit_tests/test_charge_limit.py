"""`max_total_charge_usd` caps what a pay-per-event Actor run may cost.

The developer sets it on a tool, the retriever, the client or a wrapper call, never the LLM. apify-client takes it as
a `Decimal` under `max_total_charge_usd`; going through `str` keeps a value like `0.1` exact.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from langchain_apify import ApifyActorsTool, ApifyCrawlLoader, ApifyRunActorTool, ApifySearchRetriever
from langchain_apify._client import ApifyToolsClient
from langchain_apify._utils import _charge_limit
from tests.unit_tests.conftest import SUCCEEDED_RUN


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param(0.1, Decimal('0.1'), id='float stays exact'),
        pytest.param(2, Decimal(2), id='int'),
        pytest.param(None, None, id='no limit'),
    ],
)
def test_charge_limit_converts_to_decimal(value: float | None, expected: Decimal | None) -> None:
    """The limit reaches apify-client as an exact `Decimal`, or `None` for no limit."""
    assert _charge_limit(value) == expected


def _client(max_total_charge_usd: float | None = None) -> tuple[ApifyToolsClient, MagicMock]:
    apify_client = MagicMock()
    apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
    apify_client.task.return_value.call.return_value = SUCCEEDED_RUN
    apify_client.dataset.return_value.list_items.return_value.items = []
    with patch('langchain_apify._client._create_apify_client', return_value=apify_client):
        client = ApifyToolsClient(apify_token='dummy-token', max_total_charge_usd=max_total_charge_usd)
    return client, apify_client


def test_client_passes_the_limit_to_actor_runs() -> None:
    """Every Actor run the client starts, including the domain methods, carries the limit."""
    client, apify_client = _client(max_total_charge_usd=0.1)

    client.run_actor('apify/hello-world')
    client.google_search('apify')

    for call in apify_client.actor.return_value.call.call_args_list:
        assert call.kwargs['max_total_charge_usd'] == Decimal('0.1')


def test_client_passes_the_limit_to_task_runs() -> None:
    """Task runs carry the limit too."""
    client, apify_client = _client(max_total_charge_usd=0.1)

    client.run_task('user/my-task')

    assert apify_client.task.return_value.call.call_args.kwargs['max_total_charge_usd'] == Decimal('0.1')


def test_client_without_a_limit_leaves_the_call_unchanged() -> None:
    """Without a limit, the run call gets no `max_total_charge_usd` argument at all."""
    client, apify_client = _client()

    client.run_actor('apify/hello-world')
    client.run_task('user/my-task')

    assert 'max_total_charge_usd' not in apify_client.actor.return_value.call.call_args.kwargs
    assert 'max_total_charge_usd' not in apify_client.task.return_value.call.call_args.kwargs


def test_generic_tool_hands_the_limit_to_its_client() -> None:
    """A generic tool creates its client with the developer's limit."""
    with patch('langchain_apify.tools.base.ApifyToolsClient') as client_cls:
        ApifyRunActorTool(apify_token='dummy-token', max_total_charge_usd=0.5)

    assert client_cls.call_args.kwargs['max_total_charge_usd'] == 0.5


def test_generic_tool_does_not_let_the_llm_set_the_limit() -> None:
    """The limit is a tool field, not a tool argument, so the LLM cannot raise it."""
    with patch('langchain_apify.tools.base.ApifyToolsClient'):
        tool = ApifyRunActorTool(apify_token='dummy-token', max_total_charge_usd=0.5)

    assert 'max_total_charge_usd' not in tool.args


def test_retriever_hands_the_limit_to_its_client() -> None:
    """The retriever creates its client with the developer's limit."""
    with patch('langchain_apify.retrievers.ApifyToolsClient') as client_cls:
        ApifySearchRetriever(apify_token='dummy-token', max_total_charge_usd=0.5)

    assert client_cls.call_args.kwargs['max_total_charge_usd'] == 0.5


def _actors_tool(max_total_charge_usd: float | None = None) -> tuple[ApifyActorsTool, MagicMock]:
    apify_client = MagicMock()
    build = {'actorDefinition': {'input': {'type': 'object', 'properties': {}}}}
    with (
        patch('langchain_apify.tools.actors._create_apify_client', return_value=apify_client),
        patch('langchain_apify.tools.actors._get_actor_latest_build', return_value=build),
    ):
        tool = ApifyActorsTool(
            'apify/hello-world', apify_token='dummy-token', max_total_charge_usd=max_total_charge_usd
        )
    apify_client.run.return_value.dataset.return_value.list_items.return_value.items = []
    return tool, apify_client


def test_actors_tool_passes_the_limit_to_the_run() -> None:
    """`ApifyActorsTool` passes its limit to the Actor run."""
    tool, apify_client = _actors_tool(max_total_charge_usd=0.25)

    tool._run_actor({})

    assert apify_client.actor.return_value.call.call_args.kwargs['max_total_charge_usd'] == Decimal('0.25')


def test_actors_tool_without_a_limit_leaves_the_call_unchanged() -> None:
    """Without a limit, `ApifyActorsTool` calls the Actor exactly as before."""
    tool, apify_client = _actors_tool()

    tool._run_actor({})

    apify_client.actor.return_value.call.assert_called_once_with(run_input={})


_INVALID_LIMITS = [
    pytest.param(0, id='zero'),
    pytest.param(-1, id='negative'),
    pytest.param(float('nan'), id='nan'),
    pytest.param(float('inf'), id='inf'),
]


@pytest.mark.parametrize('value', _INVALID_LIMITS)
def test_invalid_limits_are_rejected_up_front(value: float) -> None:
    """A limit that isn't a positive finite number fails when the limit is set, not when a run starts."""
    build = {'actorDefinition': {'input': {'type': 'object', 'properties': {}}}}
    with (
        patch('langchain_apify._client._create_apify_client'),
        patch('langchain_apify.tools.actors._create_apify_client'),
        patch('langchain_apify.tools.actors._get_actor_latest_build', return_value=build),
    ):
        with pytest.raises(ValueError, match='max_total_charge_usd'):
            _charge_limit(value)
        with pytest.raises(ValidationError):
            ApifyRunActorTool(apify_token='dummy-token', max_total_charge_usd=value)
        with pytest.raises(ValidationError):
            ApifySearchRetriever(apify_token='dummy-token', max_total_charge_usd=value)
        with pytest.raises(ValueError, match='max_total_charge_usd'):
            ApifyToolsClient(apify_token='dummy-token', max_total_charge_usd=value)
        with pytest.raises(ValueError, match='max_total_charge_usd'):
            ApifyCrawlLoader('https://example.com', apify_token='dummy-token', max_total_charge_usd=value)
        with pytest.raises(ValueError, match='max_total_charge_usd'):
            ApifyActorsTool('apify/hello-world', apify_token='dummy-token', max_total_charge_usd=value)


def _client_factory(created: list[MagicMock]) -> object:
    """A stand-in for `_create_apify_client` that hands out a new mock Apify client each time and records it."""

    def create(*_args: object) -> MagicMock:
        apify_client = MagicMock()
        apify_client.actor.return_value.call.return_value = SUCCEEDED_RUN
        apify_client.dataset.return_value.list_items.return_value.items = []
        created.append(apify_client)
        return apify_client

    return create


def test_a_copy_with_a_new_limit_runs_with_that_limit() -> None:
    """A copy made with `model_copy(update=...)` gets its own client, so only its runs carry the new limit."""
    created: list[MagicMock] = []
    with patch('langchain_apify._client._create_apify_client', side_effect=_client_factory(created)):
        tool = ApifyRunActorTool(apify_token='dummy-token', max_total_charge_usd=0.1)
        copy = tool.model_copy(update={'max_total_charge_usd': 0.5})

    copy.invoke({'actor_id': 'apify/hello-world'})
    tool.invoke({'actor_id': 'apify/hello-world'})

    original_client, copy_client = created
    assert copy_client.actor.return_value.call.call_args.kwargs['max_total_charge_usd'] == Decimal('0.5')
    assert original_client.actor.return_value.call.call_args.kwargs['max_total_charge_usd'] == Decimal('0.1')


def test_a_retriever_copy_with_a_new_limit_runs_with_that_limit() -> None:
    """The retriever's copy gets its own client with the new limit too."""
    created: list[MagicMock] = []
    with patch('langchain_apify._client._create_apify_client', side_effect=_client_factory(created)):
        retriever = ApifySearchRetriever(apify_token='dummy-token', max_total_charge_usd=0.1)
        copy = retriever.model_copy(update={'max_total_charge_usd': 0.5})

    copy.invoke('apify')

    original_client, copy_client = created
    assert copy_client.actor.return_value.call.call_args.kwargs['max_total_charge_usd'] == Decimal('0.5')
    original_client.actor.return_value.call.assert_not_called()


def test_the_limit_cannot_be_changed_by_assignment() -> None:
    """Assigning a new limit would leave the client with the old one, so it raises instead."""
    with patch('langchain_apify._client._create_apify_client'):
        tool = ApifyRunActorTool(apify_token='dummy-token', max_total_charge_usd=0.1)
        retriever = ApifySearchRetriever(apify_token='dummy-token', max_total_charge_usd=0.1)

    with pytest.raises(ValidationError):
        tool.max_total_charge_usd = 0.5
    with pytest.raises(ValidationError):
        retriever.max_total_charge_usd = 0.5  # ty: ignore[invalid-assignment] - the test checks that this raises


def test_crawl_loader_hands_the_limit_to_its_client() -> None:
    """`ApifyCrawlLoader` creates its client with the developer's limit."""
    with patch('langchain_apify.document_loaders.ApifyToolsClient') as client_cls:
        ApifyCrawlLoader('https://example.com', apify_token='dummy-token', max_total_charge_usd=0.5)

    assert client_cls.call_args.kwargs['max_total_charge_usd'] == 0.5
