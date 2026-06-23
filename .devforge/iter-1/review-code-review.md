VERDICT: FAIL

---

## Findings

### F1 — `core.py` error handling misses `ValueError` (confidence 92)

**Severity: BUG**

`langchain_apify/tools/core.py` imports only `_ApifyGenericTool` from `base.py` (line 22) and uses bare `except RuntimeError` in all six `_run` methods (lines 166, 216, 280, 332, 392, 456). The shared error tuple `_TOOL_RUN_ERRORS = (RuntimeError, ValueError)` is defined in `base.py` specifically to be used in every `_run`, and both `search.py` and `social.py` import and use it correctly (e.g., `search.py` line 32: `from langchain_apify.tools.base import _TOOL_RUN_ERRORS, _ApifyGenericTool`).

A `ValueError` raised by client-side input validation (e.g., a malformed `actor_id`, or a bad dataset ID) will propagate uncaught from any core tool, bypassing `handle_tool_error` and the `ToolException` wrapping that all other tools apply. This is a real behavioural gap that will surface in production when an LLM passes invalid input to a core tool.

Fix: add `_TOOL_RUN_ERRORS` to the import in `core.py` and replace all `except RuntimeError` with `except _TOOL_RUN_ERRORS` (6 sites).

---

### F2 — `tools/__init__.py` `__all__` exposes private internals (confidence 92)

**Severity: EXPORT HYGIENE BUG**

`langchain_apify/tools/__init__.py` lines 119–127 explicitly declare seven leading-underscore names in `__all__`:

```
'_DESC_DATASET_ITEMS_LIMIT',
'_DESC_MEMORY_MBYTES',
'_DESC_RUN_TIMEOUT_SECS',
'_TOOL_RUN_ERRORS',
'_VALID_MEMORY_MBYTES',
'_ApifyGenericTool',
'_iso',
'_run_meta',
```

Any `from langchain_apify.tools import *` will pull these into the consumer's namespace. This creates an implied public API for implementation details that are explicitly marked private by their leading underscore. It also causes IDE autocompletion and static-analysis tools to surface them as public exports. The top-level `langchain_apify/__init__.py` correctly omits all of them, creating an inconsistency where the subpackage advertises a broader (and wrong) surface than the top-level package.

Fix: remove all seven `_`-prefixed names from `langchain_apify/tools/__init__.py`'s `__all__`.

---

### F3 — `test_field_description_cap_matches_base_class_default` is a tautology (confidence 90)

**Severity: TEST CORRECTNESS**

The test at `tests/unit_tests/test_clamp_descriptions.py` lines 68–84 is designed to catch drift between advertised cap values in field descriptions and the actual defaults on `_ApifyGenericTool`. However, both sides of the assertion are derived from the same `_constants.py` constants at import time:

- The input schema `Field(description=...)` strings in `core.py` are f-strings: `f'... (clamped to {_MAX_TIMEOUT_SECS_CAP} max)'` (core.py lines 34, 37, 39, 57, 79 — all import from `_constants`).
- The base class field defaults in `base.py` are: `Field(default=_MAX_TIMEOUT_SECS_CAP, ...)` (base.py lines 83, 86, 89, 92 — same `_constants` imports).
- The test extracts the number from the description string (`advertised_cap`) and compares it to `_ApifyGenericTool.model_fields[cap_field].default` (`actual_cap`).

At runtime, both sides resolve to the same value from `_constants.py`. If a developer changes a description string to a hardcoded literal (e.g., `"clamped to 300 max"`) while the constant remains 600, the test will catch it only if the base class field default diverges too — which it will not if the constant is unchanged. The test thus cannot detect description-string drift that doesn't also corrupt the base class default; it only detects the case where both are wrong by the same amount.

The companion test `test_field_description_carries_cap_text` (lines 57–65) has the same tautology but also verifies the phrase exists at all, which does provide some marginal coverage. Neither test independently verifies the description number against a separately-declared ground truth.

This makes the 15 test instances in `test_field_description_cap_matches_base_class_default` non-protective against the drift scenario they claim to catch.

---

### F4 — `APIFY_API_TOKEN` env-var deprecation path: no `DeprecationWarning` emitted (confidence 92)

**Severity: BUG / SPEC VIOLATION**

`_utils.py` line 67:
```python
return os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')
```

When a user sets only `APIFY_API_TOKEN` in their environment (the old canonical name), `_resolve_apify_token()` returns the value silently with no `DeprecationWarning`. The deprecation warning machinery (`_resolve_deprecated_token`, `_resolve_deprecated_token_values`) is only triggered when `apify_api_token=` is passed as an explicit constructor keyword argument.

`DEVELOPMENT.md` and the task description both state that `APIFY_API_TOKEN` is "kept as a deprecated fallback" and "emits a DeprecationWarning." The env-var path does not fulfil this contract. Existing deployments that rely solely on the `APIFY_API_TOKEN` environment variable will continue to work silently with no migration signal.

`tests/unit_tests/test_deprecated_token_alias.py` has no test for the env-var deprecation path (grep of the file for `APIFY_API_TOKEN` and `APIFY_TOKEN` strings is empty — all tests use constructor keyword arguments only).

---

### F5 — `ApifyRAGWebBrowserTool` passes raw `self.max_timeout_secs` instead of clamping (confidence 90)

**Severity: CONSISTENCY / MINOR BUG**

`search.py` line 329: `timeout_secs=self.max_timeout_secs` is passed directly to `self._client.rag_web_search(...)`. The `ApifyRAGWebBrowserInput` schema has no `timeout_secs` field (so the LLM cannot influence it), but the raw field value `self.max_timeout_secs` is the developer-configured ceiling, not a clamped value. Every other tool passes LLM-supplied `timeout_secs` through `self._clamp_timeout(timeout_secs)`. While in this particular case there is no user-supplied value to clamp, the inconsistency means the ceiling is passed as the timeout unconditionally rather than using it as an upper bound. If a developer lowers `max_timeout_secs` to, say, 60 seconds, they get exactly 60 seconds as the timeout, which is the intended behaviour — but a value of 700 (above the platform cap) would be passed through unclamped. Using `self._clamp_timeout(self.max_timeout_secs)` would be internally consistent.

All other tools that lack a `timeout_secs` schema field and pass `self.max_timeout_secs` directly (`ApifyGoogleMapsTool`, `ApifyYouTubeScraperTool`, `ApifyInstagramScraperTool`, etc.) have the same pattern, so this is a consistent deviation, not a one-off. The root cause is the design choice to omit the field, which the existing tools replicate uniformly.

---

### F6 — `core.py` does not import or use `_TOOL_RUN_ERRORS` — inconsistency with search/social (already covered in F1 above; no separate entry)

---

## Summary table

| # | File | Lines | Severity | Issue | Confidence |
|---|------|-------|----------|-------|------------|
| F1 | `langchain_apify/tools/core.py` | 22, 166, 216, 280, 332, 392, 456 | BUG | All 6 `_run` methods catch only `RuntimeError`; `ValueError` from input validation escapes unwrapped, bypassing `ToolException` handling that `search.py` and `social.py` correctly apply | 92 |
| F2 | `langchain_apify/tools/__init__.py` | 119–127 | EXPORT HYGIENE | `__all__` explicitly exports 7 private (`_`-prefixed) symbols: `_DESC_DATASET_ITEMS_LIMIT`, `_DESC_MEMORY_MBYTES`, `_DESC_RUN_TIMEOUT_SECS`, `_TOOL_RUN_ERRORS`, `_VALID_MEMORY_MBYTES`, `_ApifyGenericTool`, `_iso`, `_run_meta` — leaking implementation details as implied public API | 92 |
| F3 | `tests/unit_tests/test_clamp_descriptions.py` | 68–84 | TEST CORRECTNESS | `test_field_description_cap_matches_base_class_default` is a tautology: both sides resolve from the same `_constants.py` values at import time; the test cannot detect drift where a description is changed to a hardcoded wrong number while the base class field default remains correct | 90 |
| F4 | `langchain_apify/_utils.py` | 67 | BUG / SPEC VIOLATION | `_resolve_apify_token()` silently consumes `APIFY_API_TOKEN` env var with no `DeprecationWarning`; no test covers this env-var path; docs claim the warning is emitted | 92 |
| F5 | `langchain_apify/tools/search.py` | 329 | CONSISTENCY | `ApifyRAGWebBrowserTool` passes `self.max_timeout_secs` raw to the client; consistent with other no-schema-field tools but diverges from the clamp-first pattern; affects all similar tools in search/social | 90 |

---

## Reviewer notes

- **Export completeness (top-level):** All 19 tool classes, 3 convenience lists, `ApifySearchRetriever`, `ApifyCrawlLoader`, and all 3 legacy names (`ApifyDatasetLoader`, `ApifyWrapper`, `ApifyActorsTool`) are correctly exported from `langchain_apify/__init__.py` and in `__all__`. No missing exports at the top-level package.
- **`test_deprecated_token_alias.py` fixture:** `mock_apify_client` is defined in `tests/unit_tests/conftest.py` (line 36) — Subagent A's concern was a false alarm; the fixture is reachable.
- **Tautology in `test_field_description_carries_cap_text` (lines 57–65):** This companion test has the same tautology root cause as F3 but provides marginal value (confirms the phrase exists) and is a less critical finding. Not promoted to a separate finding.
- **`ApifyWebCrawlerTool` uses `_clamp_items` for `max_crawl_pages`:** Confidence 82 — architecturally awkward conflation of page-count and item-count limits; omitted as it does not cross the FAIL threshold on its own but is worth a follow-up design question.
