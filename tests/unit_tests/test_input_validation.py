"""Unit tests for Pydantic input schema validation.

Covers: missing required fields, wrong types, schema-vs-clamping boundary,
string edge cases, empty/invalid IDs, malformed dicts, boundaries.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_apify import ApifyDatasetLoader
from langchain_apify.tools import (
    ApifyActorsTool,
    ApifyGetDatasetItemsInput,
    ApifyRunActorInput,
    ApifyRunTaskInput,
    ApifyScrapeUrlInput,
)


# ---------------------------------------------------------------------------
# Required vs optional field validation
# ---------------------------------------------------------------------------


class TestMissingRequiredFields:
    def test_actor_id_required(self) -> None:
        with pytest.raises(ValidationError, match='actor_id'):
            ApifyRunActorInput()  

    def test_dataset_id_required(self) -> None:
        with pytest.raises(ValidationError, match='dataset_id'):
            ApifyGetDatasetItemsInput() 

    def test_url_required(self) -> None:
        with pytest.raises(ValidationError, match='url'):
            ApifyScrapeUrlInput()  

    def test_task_id_required(self) -> None:
        with pytest.raises(ValidationError, match='task_id'):
            ApifyRunTaskInput() 


# ---------------------------------------------------------------------------
# Pydantic type validation (wrong types passed to fields)
# ---------------------------------------------------------------------------


class TestWrongTypes:
    def test_actor_id_int_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApifyRunActorInput(actor_id=12345)

    def test_dataset_id_none_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApifyGetDatasetItemsInput(dataset_id=None) 

    def test_limit_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApifyGetDatasetItemsInput(dataset_id='x', limit='ten')      

    def test_run_input_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApifyRunActorInput(actor_id='x', run_input='not a dict')

    def test_task_input_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApifyRunTaskInput(task_id='t', task_input=[1, 2, 3])


# ---------------------------------------------------------------------------
# Negative offset values
# ---------------------------------------------------------------------------


class TestOffsetBoundaries:
    def test_negative_offset_accepted_by_schema(self) -> None:
        """Pydantic does not reject negative offset; API behaviour is undefined."""
        model = ApifyGetDatasetItemsInput(dataset_id='x', offset=-1)
        assert model.offset == -1

    def test_zero_offset(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='x', offset=0)
        assert model.offset == 0

    def test_huge_offset(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='x', offset=999999)
        assert model.offset == 999999


# ---------------------------------------------------------------------------
# Empty/invalid ID strings for all fields
# ---------------------------------------------------------------------------


class TestEmptyStringBoundaries:
    def test_empty_actor_id(self) -> None:
        model = ApifyRunActorInput(actor_id='')
        assert model.actor_id == ''

    def test_empty_dataset_id(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='')
        assert model.dataset_id == ''

    def test_empty_task_id(self) -> None:
        model = ApifyRunTaskInput(task_id='')
        assert model.task_id == ''

    def test_empty_url(self) -> None:
        model = ApifyScrapeUrlInput(url='')
        assert model.url == ''


# ---------------------------------------------------------------------------
# Empty run_input dict handling
# ---------------------------------------------------------------------------


class TestRunInputDictHandling:
    def test_empty_dict_accepted(self) -> None:
        model = ApifyRunActorInput(actor_id='x', run_input={})
        assert model.run_input == {}

    def test_none_run_input_accepted(self) -> None:
        model = ApifyRunActorInput(actor_id='x', run_input=None)
        assert model.run_input is None

    def test_none_task_input_accepted(self) -> None:
        model = ApifyRunTaskInput(task_id='t', task_input=None)
        assert model.task_input is None

    def test_empty_task_input_accepted(self) -> None:
        model = ApifyRunTaskInput(task_id='t', task_input={})
        assert model.task_input == {}


# ---------------------------------------------------------------------------
# Malformed dict structure for run_input / task_input
# ---------------------------------------------------------------------------


class TestMalformedDictStructure:
    def test_deeply_nested_run_input(self) -> None:
        nested = {'a': {'b': {'c': {'d': [1, 2, {'e': True}]}}}}
        model = ApifyRunActorInput(actor_id='x', run_input=nested)
        assert model.run_input == nested

    def test_run_input_with_none_values(self) -> None:
        model = ApifyRunActorInput(actor_id='x', run_input={'key': None, 'list': [None]})
        assert model.run_input['key'] is None

    def test_run_input_with_mixed_types(self) -> None:
        mixed = {'str': 'a', 'int': 1, 'float': 1.5, 'bool': True, 'list': [], 'dict': {}}
        model = ApifyRunActorInput(actor_id='x', run_input=mixed)
        assert model.run_input == mixed


# ---------------------------------------------------------------------------
# Invalid JSON string input for ApifyActorsTool.run_input
# ---------------------------------------------------------------------------


class TestApifyActorsToolRunInput:
    """The legacy ApifyActorsTool accepts run_input as str or dict."""

    @pytest.fixture
    def tool(self) -> ApifyActorsTool:
        with (
            patch.object(ApifyActorsTool, '_create_description', return_value='desc'),
            patch.object(ApifyActorsTool, '_build_tool_args_schema_model') as mock_schema,
        ):
            from pydantic import BaseModel

            class DummyModel(BaseModel):
                run_input: str | dict

            mock_schema.return_value = DummyModel
            return ApifyActorsTool(actor_id='a/b', apify_api_token='dummy')

    def test_valid_json_string_accepted(self, tool: ApifyActorsTool) -> None:
        with patch.object(ApifyActorsTool, '_run_actor', return_value=[]):
            result = tool._run('{"key": "value"}')
            # Should not raise; parsed internally

    def test_dict_input_accepted(self, tool: ApifyActorsTool) -> None:
        with patch.object(ApifyActorsTool, '_run_actor', return_value=[]):
            result = tool._run({'key': 'value'})

    def test_invalid_json_string_raises(self, tool: ApifyActorsTool) -> None:
        with patch.object(ApifyActorsTool, '_run_actor', return_value=[]):
            with pytest.raises(Exception):
                tool._run('not valid json {{{')


# ---------------------------------------------------------------------------
# String length boundaries
# ---------------------------------------------------------------------------


class TestStringLengthBoundaries:
    def test_very_long_actor_id(self) -> None:
        long_id = 'a' * 5000
        model = ApifyRunActorInput(actor_id=long_id)
        assert len(model.actor_id) == 5000

    def test_very_long_dataset_id(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='x' * 5000)
        assert len(model.dataset_id) == 5000

    def test_very_long_url(self) -> None:
        model = ApifyScrapeUrlInput(url='https://x.com/' + 'a' * 5000)
        assert len(model.url) > 5000

    def test_very_long_task_id(self) -> None:
        model = ApifyRunTaskInput(task_id='t' * 5000)
        assert len(model.task_id) == 5000


# ---------------------------------------------------------------------------
# run_input dict size boundaries
# ---------------------------------------------------------------------------


class TestRunInputSizeBoundaries:
    def test_large_run_input_accepted(self) -> None:
        large = {f'key_{i}': f'value_{i}' for i in range(1000)}
        model = ApifyRunActorInput(actor_id='x', run_input=large)
        assert len(model.run_input) == 1000

    def test_large_nested_run_input(self) -> None:
        large = {'items': [{'id': i, 'data': 'x' * 100} for i in range(500)]}
        model = ApifyRunActorInput(actor_id='x', run_input=large)
        assert len(model.run_input['items']) == 500


# ---------------------------------------------------------------------------
# Unicode/special character boundaries
# ---------------------------------------------------------------------------


class TestUnicodeAndSpecialChars:
    def test_unicode_actor_id(self) -> None:
        model = ApifyRunActorInput(actor_id='user/actor-日本語-émojis')
        assert '日本語' in model.actor_id

    def test_special_chars_in_dataset_id(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='ds-with-dashes_and_underscores')
        assert model.dataset_id == 'ds-with-dashes_and_underscores'

    def test_url_with_unicode_path(self) -> None:
        model = ApifyScrapeUrlInput(url='https://example.com/café/日本語')
        assert 'café' in model.url

    def test_special_chars_in_task_id(self) -> None:
        model = ApifyRunTaskInput(task_id='user~my-task_v2')
        assert model.task_id == 'user~my-task_v2'


# ---------------------------------------------------------------------------
# URL format validation boundaries
# ---------------------------------------------------------------------------


class TestUrlFormatBoundaries:
    def test_malformed_url(self) -> None:
        model = ApifyScrapeUrlInput(url='not-a-url')
        assert model.url == 'not-a-url'

    def test_ftp_scheme(self) -> None:
        model = ApifyScrapeUrlInput(url='ftp://files.example.com/data')
        assert model.url.startswith('ftp://')

    def test_url_with_query_params(self) -> None:
        model = ApifyScrapeUrlInput(url='https://x.com/page?q=hello&lang=en#section')
        assert '?' in model.url and '#' in model.url

    def test_url_with_port(self) -> None:
        model = ApifyScrapeUrlInput(url='http://localhost:8080/api')
        assert ':8080' in model.url


# ---------------------------------------------------------------------------
# ApifyDatasetLoader input validation
# ---------------------------------------------------------------------------


class TestApifyDatasetLoaderValidation:
    def _dummy_fn(self, item: dict) -> Document:
        return Document(page_content=str(item))

    def test_missing_dataset_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('APIFY_API_TOKEN', 'dummy')
        with pytest.raises((TypeError, ValidationError)):
            ApifyDatasetLoader(
                dataset_id=None,  # type: ignore[arg-type]
                dataset_mapping_function=self._dummy_fn,
            )

    def test_missing_mapping_function_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('APIFY_API_TOKEN', 'dummy')
        with pytest.raises((TypeError, ValidationError)):
            ApifyDatasetLoader(
                dataset_id='ds-1',
                dataset_mapping_function=None,  # type: ignore[arg-type]
            )

    def test_empty_dataset_id_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string passes validation; API will reject it."""
        monkeypatch.setenv('APIFY_API_TOKEN', 'dummy')
        with patch('langchain_apify.document_loaders._create_apify_client'):
            loader = ApifyDatasetLoader(dataset_id='', dataset_mapping_function=self._dummy_fn)
        assert loader.dataset_id == ''

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
        monkeypatch.delenv('APIFY_TOKEN', raising=False)
        with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
            ApifyDatasetLoader(dataset_id='ds-1', dataset_mapping_function=self._dummy_fn)


# ---------------------------------------------------------------------------
# Schema does not enforce range (clamping handles it)
# ---------------------------------------------------------------------------


class TestSchemaDoesNotEnforceRange:
    def test_negative_timeout_accepted(self) -> None:
        model = ApifyRunActorInput(actor_id='x', timeout_secs=-1)
        assert model.timeout_secs == -1

    def test_negative_limit_accepted(self) -> None:
        model = ApifyGetDatasetItemsInput(dataset_id='x', limit=-1)
        assert model.limit == -1
