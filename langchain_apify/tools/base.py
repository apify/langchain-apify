"""Shared base for generic Apify tools.

Hosts the :class:`_ApifyGenericTool` base class, the JSON envelope helper,
run-metadata helpers, the developer-controlled clamp methods, and the shared
``_TOOL_RUN_ERRORS`` contract used by every tool ``_run``.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, PrivateAttr, SecretStr, model_validator

from langchain_apify._client import ApifyToolsClient
from langchain_apify._constants import (
    _MAX_CRAWL_DEPTH_CAP,
    _MAX_ITEMS_CAP,
    _MAX_MEMORY_MBYTES_CAP,
    _MAX_TIMEOUT_SECS_CAP,
)
from langchain_apify._error_messages import _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
from langchain_apify._utils import (
    _apify_token_secret_factory,
    _resolve_deprecated_token_values,
)

# Apify accepts memory_mbytes only as one of these power-of-2 values.
# https://docs.apify.com/api/v2/act-runs-post
_VALID_MEMORY_MBYTES: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Errors a tool ``_run`` converts into a ``ToolException``: ``RuntimeError`` from
# a failed/empty Actor run, ``ValueError`` from client-side input validation.
_TOOL_RUN_ERRORS: tuple[type[Exception], ...] = (RuntimeError, ValueError)


def _iso(value: str | datetime | None) -> str | None:
    """Coerce a possible ``datetime`` to an ISO-8601 string."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _run_meta(run: dict) -> dict:
    """Extract a compact metadata dict from an Apify run-details dict."""
    return {
        'run_id': run.get('id'),
        'status': run.get('status'),
        'dataset_id': run.get('defaultDatasetId'),
        'started_at': _iso(run.get('startedAt')),
        'finished_at': _iso(run.get('finishedAt')),
    }


# ---------------------------------------------------------------------------
# Shared base for generic tools
# ---------------------------------------------------------------------------


class _ApifyGenericTool(BaseTool):  # type: ignore[override]
    """Shared base for all generic Apify tools.

    Handles ``ApifyToolsClient`` creation, sets ``handle_tool_error``,
    and defines developer-controlled safety limits that clamp values the
    LLM may provide at invocation time.

    Subclasses only need to declare ``name``, ``description``,
    ``args_schema``, and ``_run()``.
    """

    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True

    apify_token: SecretStr | None = Field(
        default_factory=_apify_token_secret_factory,
        description='Apify API token. Falls back to the APIFY_TOKEN environment variable when None.',
        exclude=True,
        repr=False,
    )
    max_timeout_secs: int = Field(
        default=_MAX_TIMEOUT_SECS_CAP, description='Upper bound for timeout_secs the LLM may request.'
    )
    max_memory_mbytes: int = Field(
        default=_MAX_MEMORY_MBYTES_CAP, description='Upper bound for memory_mbytes the LLM may request.'
    )
    max_items: int = Field(
        default=_MAX_ITEMS_CAP, description='Upper bound for limit / dataset_items_limit the LLM may request.'
    )
    max_crawl_depth: int = Field(
        default=_MAX_CRAWL_DEPTH_CAP, description='Upper bound for max_crawl_depth the LLM may request.'
    )

    _client: ApifyToolsClient = PrivateAttr()

    @model_validator(mode='before')
    @classmethod
    def _handle_deprecated_apify_api_token(cls, values: dict) -> dict:
        return _resolve_deprecated_token_values(values)

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        if self.apify_token is None:
            msg = _ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET
            raise ValueError(msg)
        self._client = ApifyToolsClient(apify_token=self.apify_token.get_secret_value())
        super().model_post_init(context)

    def _clamp_timeout(self, value: int) -> int:
        return max(1, min(value, self.max_timeout_secs))

    def _clamp_memory(self, value: int | None) -> int | None:
        if value is None or value <= 0:
            return None
        clamped = max(128, min(value, self.max_memory_mbytes))
        idx = bisect.bisect_left(_VALID_MEMORY_MBYTES, clamped)
        # If snap-up exceeds cap, use largest valid at-or-below cap
        if idx >= len(_VALID_MEMORY_MBYTES) or _VALID_MEMORY_MBYTES[idx] > self.max_memory_mbytes:
            idx = bisect.bisect_right(_VALID_MEMORY_MBYTES, self.max_memory_mbytes) - 1
        # Misconfigured cap below the platform minimum, return the minimum.
        return _VALID_MEMORY_MBYTES[max(idx, 0)]

    def _clamp_items(self, value: int) -> int:
        return max(1, min(value, self.max_items))

    def _clamp_depth(self, value: int) -> int:
        # Floor at 0 (a depth of 0 means "only crawl the seed URL").
        return max(0, min(value, self.max_crawl_depth))

    @staticmethod
    def _envelope(run: dict | None, items: list) -> str:
        """Serialise the standard ``{"run": ..., "items": ...}`` tool envelope.

        ``run`` is a raw Apify run-details dict (passed through :func:`_run_meta`)
        or ``None`` for dataset-only tools. ``default=str`` coerces non-JSON-native
        values (e.g. ``datetime`` objects from the ``clean=True`` deserialiser) so
        serialisation never raises ``TypeError``.
        """
        return json.dumps(
            {'run': _run_meta(run) if run is not None else None, 'items': items},
            default=str,
        )
