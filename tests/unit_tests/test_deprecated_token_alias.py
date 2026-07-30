"""Tests for the ``apify_api_token`` → ``apify_token`` deprecation alias.

Every public class that accepts an Apify token is covered:
  - ApifyToolsClient
  - ApifyDatasetLoader
  - ApifyWrapper
  - ApifyActorsTool
  - _ApifyGenericTool  (via ApifyRunActorTool)

For each class, the matrix is:
  1. ``apify_token``      → works, NO warning
  2. ``apify_api_token``  → works, emits DeprecationWarning ("deprecated, use apify_token")
  3. both specified       → ``apify_token`` wins, emits DeprecationWarning
                            ("ignoring the deprecated 'apify_api_token'")
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from apify_client._types import ListPage
from apify_client.clients import DatasetClient
from langchain_core.documents import Document
from pydantic import BaseModel

from langchain_apify import ApifyDatasetLoader, ApifyWrapper
from langchain_apify._client import ApifyToolsClient
from langchain_apify._utils import _resolve_apify_token
from langchain_apify.tools import ApifyActorsTool, ApifyRunActorTool

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_LIST_PAGE = ListPage(data={'items': []})


def _noop_mapping(item: dict) -> Document:
    return Document(page_content=item.get('text', ''))


# ---------------------------------------------------------------------------
# ApifyToolsClient
# ---------------------------------------------------------------------------


class TestApifyToolsClientTokenAlias:
    """Token-alias tests for :class:`ApifyToolsClient`."""

    def test_apify_token_no_warning(self, mock_apify_client: MagicMock) -> None:
        with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                c = ApifyToolsClient(apify_token='new-style')
            assert len(w) == 0
            assert c._client is mock_apify_client

    def test_apify_api_token_emits_warning(self, mock_apify_client: MagicMock) -> None:
        with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                c = ApifyToolsClient(apify_api_token='legacy-style')
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'apify_api_token' in str(w[0].message)
            assert c._client is mock_apify_client

    def test_both_specified_uses_apify_token(self, mock_apify_client: MagicMock) -> None:
        """When both are given, ``apify_token`` wins and the user is warned."""
        with patch('langchain_apify._client._create_apify_client', return_value=mock_apify_client) as mock_create:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ApifyToolsClient(apify_token='primary', apify_api_token='ignored')
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'ignoring' in str(w[0].message)
            # apify_token was passed to _create_apify_client, not apify_api_token
            assert mock_create.call_args.args[1] == 'primary'


# ---------------------------------------------------------------------------
# ApifyDatasetLoader
# ---------------------------------------------------------------------------


class TestApifyDatasetLoaderTokenAlias:
    """Token-alias tests for :class:`ApifyDatasetLoader`."""

    def test_apify_token_no_warning(self) -> None:
        with patch.object(DatasetClient, 'list_items', return_value=_EMPTY_LIST_PAGE):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                loader = ApifyDatasetLoader(
                    dataset_id='d',
                    dataset_mapping_function=_noop_mapping,
                    apify_token='new-style',
                )
            assert len(w) == 0
            assert loader.apify_token is not None

    def test_apify_api_token_emits_warning(self) -> None:
        with patch.object(DatasetClient, 'list_items', return_value=_EMPTY_LIST_PAGE):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                loader = ApifyDatasetLoader(
                    dataset_id='d',
                    dataset_mapping_function=_noop_mapping,
                    apify_api_token='legacy-style',
                )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'apify_api_token' in str(w[0].message)
            assert loader.apify_token is not None

    def test_both_specified_uses_apify_token(self) -> None:
        """When both are given, ``apify_token`` wins and the user is warned."""
        with patch.object(DatasetClient, 'list_items', return_value=_EMPTY_LIST_PAGE):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                loader = ApifyDatasetLoader(
                    dataset_id='d',
                    dataset_mapping_function=_noop_mapping,
                    apify_token='primary',
                    apify_api_token='ignored',
                )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'ignoring' in str(w[0].message)
            assert loader.apify_token is not None
            assert loader.apify_token.get_secret_value() == 'primary'


# ---------------------------------------------------------------------------
# ApifyWrapper
# ---------------------------------------------------------------------------


class TestApifyWrapperTokenAlias:
    """Token-alias tests for :class:`ApifyWrapper`."""

    def test_apify_token_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            wrapper = ApifyWrapper(apify_token='new-style')
        assert len(w) == 0
        assert wrapper.apify_token is not None

    def test_apify_api_token_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            wrapper = ApifyWrapper(apify_api_token='legacy-style')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert 'apify_api_token' in str(w[0].message)
        assert wrapper.apify_token is not None

    def test_both_specified_uses_apify_token(self) -> None:
        """When both are given, ``apify_token`` wins and the user is warned."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            wrapper = ApifyWrapper(apify_token='primary', apify_api_token='ignored')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert 'ignoring' in str(w[0].message)
        assert wrapper.apify_token is not None
        assert wrapper.apify_token.get_secret_value() == 'primary'


# ---------------------------------------------------------------------------
# ApifyActorsTool
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    run_input: str


class TestApifyActorsToolTokenAlias:
    """Token-alias tests for :class:`ApifyActorsTool`."""

    @staticmethod
    def _patches() -> tuple:
        """Return patch context managers that stub the network calls ApifyActorsTool makes at init."""
        return (
            patch('langchain_apify.tools.actors._get_actor_latest_build', return_value={}),
            patch.object(ApifyActorsTool, '_create_description', return_value='stub'),
            patch.object(ApifyActorsTool, '_build_tool_args_schema_model', return_value=_DummySchema),
        )

    def test_apify_token_no_warning(self) -> None:
        with ExitStack() as stack:
            for p in self._patches():
                stack.enter_context(p)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ApifyActorsTool(actor_id='apify/test', apify_token='new-style')
            assert len(w) == 0

    def test_apify_api_token_emits_warning(self) -> None:
        with ExitStack() as stack:
            for p in self._patches():
                stack.enter_context(p)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                tool = ApifyActorsTool(actor_id='apify/test', apify_api_token='legacy-style')
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'apify_api_token' in str(w[0].message)
            assert isinstance(tool, ApifyActorsTool)

    def test_both_specified_uses_apify_token(self) -> None:
        """When both are given, ``apify_token`` wins and the user is warned."""
        with ExitStack() as stack:
            for p in self._patches():
                stack.enter_context(p)
            with patch('langchain_apify.tools.actors._create_apify_client') as mock_create:
                mock_create.return_value = MagicMock()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    ApifyActorsTool(actor_id='apify/test', apify_token='primary', apify_api_token='ignored')
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert 'ignoring' in str(w[0].message)
                # ``apify_token`` was passed through to the client constructor.
                assert mock_create.call_args.args[1] == 'primary'


# ---------------------------------------------------------------------------
# _ApifyGenericTool  (tested via ApifyRunActorTool)
# ---------------------------------------------------------------------------


class TestGenericToolTokenAlias:
    """Token-alias tests for :class:`_ApifyGenericTool` (via :class:`ApifyRunActorTool`)."""

    def test_apify_token_no_warning(self) -> None:
        with patch.object(ApifyToolsClient, '__init__', return_value=None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                tool = ApifyRunActorTool(apify_token='new-style')  # type: ignore[call-arg,arg-type]
            assert len(w) == 0
            assert tool.apify_token is not None

    def test_apify_api_token_emits_warning(self) -> None:
        with patch.object(ApifyToolsClient, '__init__', return_value=None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                tool = ApifyRunActorTool(apify_api_token='legacy-style')  # type: ignore[call-arg,arg-type]
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'apify_api_token' in str(w[0].message)
            assert tool.apify_token is not None

    def test_both_specified_uses_apify_token(self) -> None:
        """When both are given, ``apify_token`` wins and the user is warned."""
        with patch.object(ApifyToolsClient, '__init__', return_value=None):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                tool = ApifyRunActorTool(apify_token='primary', apify_api_token='ignored')  # type: ignore[call-arg,arg-type]
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'ignoring' in str(w[0].message)
            assert tool.apify_token is not None
            assert tool.apify_token.get_secret_value() == 'primary'


# ---------------------------------------------------------------------------
# APIFY_API_TOKEN environment variable (deprecated alias for APIFY_TOKEN)
# ---------------------------------------------------------------------------


class TestApifyApiTokenEnvVarAlias:
    """The ``APIFY_API_TOKEN`` env var is honoured but deprecated.

    Mirrors the kwarg matrix: ``APIFY_TOKEN`` is preferred and silent,
    ``APIFY_API_TOKEN`` works but emits a ``DeprecationWarning``.
    """

    def test_apify_token_env_no_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('APIFY_TOKEN', 'new-style')
        monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            token = _resolve_apify_token()
        assert token == 'new-style'
        assert len(w) == 0

    def test_apify_api_token_env_emits_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('APIFY_TOKEN', raising=False)
        monkeypatch.setenv('APIFY_API_TOKEN', 'legacy-style')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            token = _resolve_apify_token()
        assert token == 'legacy-style'
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert 'APIFY_API_TOKEN' in str(w[0].message)

    def test_apify_token_env_takes_precedence_silently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both env vars are set, ``APIFY_TOKEN`` wins with no warning."""
        monkeypatch.setenv('APIFY_TOKEN', 'primary')
        monkeypatch.setenv('APIFY_API_TOKEN', 'ignored')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            token = _resolve_apify_token()
        assert token == 'primary'
        assert len(w) == 0

    def test_no_token_env_returns_none_without_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('APIFY_TOKEN', raising=False)
        monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            token = _resolve_apify_token()
        assert token is None
        assert len(w) == 0
