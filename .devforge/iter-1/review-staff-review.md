VERDICT: FAIL

## Summary

The implementation is structurally sound and the bulk of the new API surface is correct. Four issues prevent a clean PASS: two of them are correctness gaps (a breaking change without a CHANGELOG entry, and a gap between documented and implemented deprecation behavior), and two are design cleanliness issues in the subpackage `__all__` and an isolated timeout bypass.

---

## Findings

| # | Severity | File | Finding |
|---|----------|------|---------|
| 1 | HIGH | `langchain_apify/document_loaders.py`, `CHANGELOG.md` | **Breaking rename undocumented.** `ApifyDatasetLoader.apify_client` (public attribute) is now `_apify_client` (PrivateAttr). Any downstream code doing `loader.apify_client` will get a silent `AttributeError` at runtime. The CHANGELOG `0.1.5` section has no breaking-change entry for this rename. The `CONTRIBUTING.md` mentions `BREAKING CHANGE:` footer convention but neither the commit message nor CHANGELOG uses it. Per the design review focus ("Check whether the PR includes a CHANGELOG entry or migration note"), this is an unmitigated breaking change that must be disclosed. |
| 2 | MEDIUM | `langchain_apify/_utils.py:67`, `_verified_task.md` | **`APIFY_API_TOKEN` env var emits no `DeprecationWarning`.** The `_verified_task.md` states "Code using only `APIFY_API_TOKEN` will still work (with `DeprecationWarning`)". This is inaccurate: `_resolve_apify_token()` returns `os.getenv('APIFY_API_TOKEN')` silently with no `warnings.warn` call. The `DeprecationWarning` is only emitted for the `apify_api_token` *kwarg* path (via `_resolve_deprecated_token` and `_resolve_deprecated_token_values`). Users who set only the env var are not warned to migrate to `APIFY_TOKEN`. Either the warning must be added to `_resolve_apify_token()` or the design doc must be corrected—but the design doc is normative here, so the implementation is the gap. |
| 3 | MEDIUM | `langchain_apify/tools/__init__.py:119-127` | **Internal symbols in subpackage `__all__`.** `langchain_apify.tools.__all__` explicitly exports `_ApifyGenericTool`, `_TOOL_RUN_ERRORS`, `_VALID_MEMORY_MBYTES`, `_iso`, `_run_meta`, `_DESC_DATASET_ITEMS_LIMIT`, `_DESC_MEMORY_MBYTES`, `_DESC_RUN_TIMEOUT_SECS`. These are private implementation details (leading-underscore names). The design review focus states "No internal symbols (leading-underscore classes, `_ApifyGenericTool`, `_constants`, `_types`) should appear in `__all__`." The top-level `langchain_apify/__init__.py` is clean (none of these reach the public API via `from langchain_apify import *`), but `from langchain_apify.tools import *` does expose them, and they are now implicitly under the stability guarantee of the package. |
| 4 | LOW | `langchain_apify/tools/search.py:329,394,454,514`, `tools/social.py:217,273,329,385,454,514,578` | **Inconsistent timeout handling: `self.max_timeout_secs` used directly as `timeout_secs` in 11 tool `_run` methods.** `ApifyGoogleSearchTool` and `ApifyWebCrawlerTool` correctly accept a user-supplied `timeout_secs` in their input schemas and clamp it via `self._clamp_timeout(timeout_secs)`. All other 11 specialized tools (`ApifyRAGWebBrowserTool`, `ApifyGoogleMapsTool`, `ApifyYouTubeScraperTool`, `ApifyEcommerceScraperTool`, and all 7 social tools) pass `timeout_secs=self.max_timeout_secs` directly—the *cap ceiling*—not a clamped user value. Since these tools' input schemas have no `timeout_secs` field, the LLM cannot influence the timeout, which is a deliberate design choice. However: (a) the developer cannot configure a lower-than-cap timeout for these tools at construction time without reducing `max_timeout_secs`, and (b) passing the ceiling as the actual timeout means runs always run at maximum latency tolerance. This is a design inconsistency (some tools allow LLM timeout control, others do not) that should be explicit in the docstrings. No functional bug, but warrants a nit or documentation clarification. |

---

## Focus-area audit

### 1. `tools/base.py` — `_ApifyGenericTool`

- **Clamp monotonicity**: PASS. `_clamp_timeout` and `_clamp_items`/`_clamp_depth` use `max(floor, min(value, cap))` correctly. `_clamp_memory` uses `bisect` to snap to the next valid power-of-2; tested against boundary values (127→128, 200→256, 32769→32768, non-power-of-2 cap→snaps down, sub-minimum cap→returns 128). All behave correctly.
- **`_envelope` safety**: PASS. `_run_meta(run)` never mutates the input dict (only reads via `.get()`). `json.dumps(..., default=str)` handles `datetime` and other non-native types. Empty list and `None` for `run` both serialize correctly.
- **PrivateAttr client construction**: PASS. `model_post_init` constructs `_client` before `super().model_post_init()` and raises `ValueError` (not `AttributeError`) if token is absent.
- **Token resolution order**: PASS. `apify_token` field → `APIFY_TOKEN` env → `APIFY_API_TOKEN` env (silent) → raises `ValueError`. The kwarg deprecation path (`apify_api_token` kwarg → warning) is handled by the `model_validator(mode='before')`.

### 2. `_client.py` — `ApifyToolsClient`

- **Actor ID uniqueness/accuracy**: All 13 Actor IDs are distinct. IDs match known Apify Store slugs (`apify/website-content-crawler`, `apify/google-search-scraper`, `apify/rag-web-browser`, `compass/crawler-google-places`, `streamers/youtube-scraper`, `apify/e-commerce-scraping-tool`, `apify/instagram-scraper`, `apimaestro/linkedin-profile-posts`, `harvestapi/linkedin-profile-search`, `apimaestro/linkedin-profile-detail`, `apidojo/twitter-scraper-lite`, `clockworks/tiktok-scraper`, `apify/facebook-posts-scraper`).
- **None-in-payload handling**: PASS. All optional params (`memory_mbytes`, `only_posts_newer_than`, `country_code`, etc.) use `if x is not None: run_input[key] = x` guards before assignment. Required fields are always included.
- **Error propagation**: PASS. `_check_run_status` raises `RuntimeError` on non-SUCCEEDED status. `_TRANSPORT_EXCEPTIONS = (ApifyClientError, httpx.HTTPError)` wrapped to `RuntimeError`. Tools catch `RuntimeError`/`ValueError` from `_TOOL_RUN_ERRORS` and re-raise as `ToolException`.

### 3. `_utils.py` — falsy-default fix + `_get_actor_latest_build`

- **Falsy-default fix**: PASS. `if (value := meta.get(key_name)) is not None:` correctly preserves `0`, `False`, `""`. `default=None` (explicit null) is still pruned because `None is not None` is `False`. The old truthy-gate bug is fixed.
- **`_get_actor_latest_build` no-build edge case**: PASS for "no data" case (raises `ValueError`). Called once in `actors.py:__init__` and result passed to both `_create_description` and `_build_tool_args_schema_model`—the 2→1 fetch reduction is complete.

### 4. `document_loaders.py` — breaking rename + `ApifyCrawlLoader`

- **PrivateAttr init before use**: PASS. `_init_client` model validator (mode='after') initializes `_apify_client` before any `load()`/`lazy_load()` call can reach it. A partially-initialized loader cannot exist.
- **`ApifyCrawlLoader` BaseLoader contract**: PASS. `lazy_load` yields `Document` objects. Empty `page_content` is possible if Actor returns no markdown/text, but no silent empty-list return; items missing content still yield a `Document` with `page_content=""`.
- **Breaking change documentation**: FAIL — see Finding #1.

### 5. `__init__.py` — export completeness

- **Old exports preserved**: PASS. `ApifyDatasetLoader`, `ApifyWrapper`, `ApifyActorsTool` all present in `__all__`.
- **19 new tool classes present**: PASS. All 19 counted in `__all__`: 6 core + 6 search + 7 social. `ApifyCrawlLoader`, `ApifySearchRetriever`, `APIFY_CORE_TOOLS`, `APIFY_SEARCH_TOOLS`, `APIFY_SOCIAL_TOOLS` also present.
- **No internal symbols in top-level `__all__`**: PASS. Top-level `__init__.py` is clean.
- **Subpackage `__all__` leak**: FAIL — see Finding #3.

### 6. `test_clamp_descriptions.py` — drift protection

- **No direct constant import**: PASS. `test_field_description_cap_matches_base_class_default` reads `_ApifyGenericTool.model_fields[cap_field].default` at test time. This is a live field lookup, not a direct import of `_MAX_TIMEOUT_SECS_CAP` etc. The test would catch a drift where the model field default changes but the description string does not.
- **Cap coverage completeness**: All four cap constants (`_MAX_TIMEOUT_SECS_CAP`, `_MAX_MEMORY_MBYTES_CAP`, `_MAX_ITEMS_CAP`, `_MAX_CRAWL_DEPTH_CAP`) are covered by the 15 `_CLAMP_FIELDS` entries. PASS.

---

## Required actions before merge

1. **(HIGH)** Add a `BREAKING CHANGE:` entry to `CHANGELOG.md` under `0.1.5` documenting the `apify_client → _apify_client` rename in `ApifyDatasetLoader`, with migration guidance (`loader._apify_client` is private; use `loader.load()` / `loader.lazy_load()` instead of accessing the client directly).
2. **(MEDIUM)** Either add a `DeprecationWarning` in `_resolve_apify_token()` when `APIFY_API_TOKEN` env var is used (and `APIFY_TOKEN` is absent), or remove the claim in `_verified_task.md` / `_error_messages.py` that `APIFY_API_TOKEN` env var usage emits a warning. Recommended: add the warning so behavior matches documentation.
3. **(MEDIUM)** Remove internal symbols from `langchain_apify/tools/__init__.py`'s `__all__`. The underscore-prefixed names (`_ApifyGenericTool`, `_TOOL_RUN_ERRORS`, `_VALID_MEMORY_MBYTES`, `_iso`, `_run_meta`, `_DESC_*`) should not be in `__all__`. They can remain importable for intra-package use but should not be in the public `__all__` list.
4. **(LOW/NIT)** Document in the docstrings of `ApifyRAGWebBrowserTool`, `ApifyGoogleMapsTool`, `ApifyYouTubeScraperTool`, `ApifyEcommerceScraperTool`, and all social tools that `timeout_secs` is fixed to `max_timeout_secs` (the constructor-level cap) and is not LLM-controllable—unlike `ApifyGoogleSearchTool` and `ApifyWebCrawlerTool`.
