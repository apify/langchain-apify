"""Unit tests for _run_meta() edge cases NOT covered by test_tools.py.

Existing tests cover: datetime/string/None conversion, JSON serialization.
These add: empty dict, extra keys filtered, different statuses.
"""

from __future__ import annotations

from langchain_apify._utils import _run_meta


def test_empty_dict_returns_all_nones() -> None:
    meta = _run_meta({})
    assert meta == {
        'run_id': None,
        'status': None,
        'dataset_id': None,
        'started_at': None,
        'finished_at': None,
    }


def test_extra_api_keys_are_filtered() -> None:
    run = {
        'id': 'run-x',
        'status': 'SUCCEEDED',
        'defaultDatasetId': 'ds-x',
        'startedAt': '2025-01-01T00:00:00Z',
        'finishedAt': '2025-01-01T00:01:00Z',
        'actId': 'abc',
        'buildId': 'xyz',
        'exitCode': 0,
        'usageTotalUsd': 0.001,
    }
    meta = _run_meta(run)
    assert set(meta.keys()) == {'run_id', 'status', 'dataset_id', 'started_at', 'finished_at'}


def test_non_succeeded_status_preserved() -> None:
    for status in ('FAILED', 'TIMED-OUT', 'RUNNING', 'ABORTING'):
        meta = _run_meta({'id': 'r', 'status': status, 'defaultDatasetId': None})
        assert meta['status'] == status
