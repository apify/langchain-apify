# Fact-check — PR #35: feat: modernize langchain integration connector

Verified against working tree at `/home/user/langchain-apify` (branch `feat/modernize-langchain-integration`, HEAD 75dc0cd) and the diff at `.devforge/iter-1/diff.patch`.

---

## Claim Ledger

### 1 — 6 core tools exist: ApifyRunActorTool, ApifyGetDatasetItemsTool, ApifyRunActorAndGetDatasetTool, ApifyScrapeUrlTool, ApifyRunTaskTool, ApifyRunTaskAndGetDatasetTool

**VALID** — All six classes present in `langchain_apify/tools/core.py`; all exported from `langchain_apify/tools/__init__.py` and `langchain_apify/__init__.py`; grouped in `APIFY_CORE_TOOLS` list (line 462 of `core.py`).

---

### 2 — ApifySearchRetriever (BaseRetriever) and ApifyCrawlLoader (BaseLoader) exist

**VALID** — `ApifySearchRetriever` in `langchain_apify/retrievers.py` extends `BaseRetriever`; `ApifyCrawlLoader` in `langchain_apify/document_loaders.py` extends `BaseLoader`. Both exported from the package `__init__.py`.

---

### 3 — 6 search/crawling tools: ApifyGoogleSearchTool, ApifyWebCrawlerTool, ApifyRAGWebBrowserTool, ApifyGoogleMapsTool, ApifyYouTubeScraperTool, ApifyEcommerceScraperTool

**VALID** — All six classes present in `langchain_apify/tools/search.py`; grouped in `APIFY_SEARCH_TOOLS` list (line 522 of `search.py`).

---

### 4 — 7 social tools: Instagram, LinkedIn posts/search/detail, Twitter, TikTok, Facebook

**VALID** — Seven classes present in `langchain_apify/tools/social.py`: `ApifyInstagramScraperTool`, `ApifyLinkedInProfilePostsTool`, `ApifyLinkedInProfileSearchTool`, `ApifyLinkedInProfileDetailTool`, `ApifyTwitterScraperTool`, `ApifyTikTokScraperTool`, `ApifyFacebookPostsScraperTool`. Grouped in `APIFY_SOCIAL_TOOLS` list (line 586 of `social.py`).

---

### 5 — APIFY_TOKEN is the new primary env var; APIFY_API_TOKEN kept as deprecated alias

**VALID** — `_utils.py` line 67: `return os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')`. The diff shows old code used `APIFY_API_TOKEN` exclusively; new code elevates `APIFY_TOKEN` to primary. The `_ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET` message in `_error_messages.py` mentions `APIFY_TOKEN` first and notes `APIFY_API_TOKEN` is accepted for backwards compatibility.

**Note:** The PR description says "APIFY_TOKEN renamed, APIFY_API_TOKEN kept as deprecated alias" but the direction is actually that `APIFY_TOKEN` was already the Apify SDK standard; this PR makes it the primary and demotes `APIFY_API_TOKEN` to a deprecated alias. The technical result is the same.

---

### 6 — Package split: tools/ subpackage with base/core/actors/search/social modules

**STALE(→ corrected)** — The description says "collapsed `_actor_tools.py` + `tools.py` into a `tools/` package." There was never a separate `_actor_tools.py` in the old tree. The diff shows only `langchain_apify/tools.py` was renamed to `langchain_apify/tools/actors.py`; the new modules `base.py`, `core.py`, `search.py`, and `social.py` were added as new files. Corrected: `tools.py` (containing only `ApifyActorsTool`) was renamed to `tools/actors.py`, and four new submodules were added.

---

### 7 — _constants.py, _types.py, _error_messages.py for shared values

**VALID** — All three files present. `_constants.py` holds run/timeout/result defaults and clamp caps. `_types.py` holds Literal type aliases (`CrawlerType`, `YouTubeSearchType`, `EcommerceUrlType`, `InstagramSearchType`, `TwitterSearchMode`, `TwitterSort`, `TikTokSearchType`). `_error_messages.py` holds three error string constants. The old public names `const.py`, `error_messages.py`, `utils.py` were deleted (confirmed by diff `deleted file mode` entries).

---

### 8 — 4 clamp constants: max_timeout_secs=600, max_memory_mbytes=32768, max_items=1000, max_crawl_depth=5

**VALID** — `_constants.py` lines 35–38:
- `_MAX_TIMEOUT_SECS_CAP = 600`
- `_MAX_MEMORY_MBYTES_CAP = 32768`
- `_MAX_ITEMS_CAP = 1000`
- `_MAX_CRAWL_DEPTH_CAP = 5`

These are mirrored as `Field(default=...)` on `_ApifyGenericTool` (`tools/base.py` lines 83–93).

---

### 9 — New test file test_clamp_descriptions.py with 30 parametrised tests

**VALID** — File exists at `tests/unit_tests/test_clamp_descriptions.py`. It has `_CLAMP_FIELDS` list with 15 entries and 2 `@pytest.mark.parametrize` test functions, yielding exactly 30 test instances (15 × 2). The 2 test functions are `test_field_description_carries_cap_text` and `test_field_description_cap_matches_base_class_default`.

---

### 10 — ApifyWrapper.(a)call_actor(_task) forwards apify_token to ApifyDatasetLoader

**VALID** — `wrappers.py` lines 151–155 and 197–202: all four methods (`call_actor`, `acall_actor`, `call_actor_task`, `acall_actor_task`) construct `ApifyDatasetLoader(..., apify_token=self.apify_token)`. The diff confirms the old code did not pass `apify_token` to the loader.

---

### 11 — _prune_actor_input_schema bug fix for falsy defaults (0, false, "")

**VALID** — `_utils.py` line 145: `if (value := meta.get(key_name)) is not None:`. The old code (visible in diff) used a truthy check `if (value := meta.get(key_name)):` which would silently drop `0`, `false`, and `""` defaults from the schema. The fix uses `is not None` which preserves all falsy-but-present values.

---

### 12 — ApifyActorsTool.__init__ reduced from 2 fetches to 1

**VALID** — The diff (lines ~2375 and ~2403) shows the old `_create_description()` called `get_actor_latest_build()` internally, and `_build_tool_args_schema_model()` also called `get_actor_latest_build()` internally — two separate network fetches. The new `__init__` calls `_get_actor_latest_build()` once (line 100 of `actors.py`) and passes the result `build` dict to both methods as an argument. Net reduction: 2 network calls → 1.

---

### 13 — ApifyDatasetLoader.apify_client changed to PrivateAttr

**VALID** — `document_loaders.py` line 66: `_apify_client: ApifyClient = PrivateAttr()`. The diff confirms the old name was `apify_client` (a regular public Pydantic field); it is now `_apify_client` (private, leading underscore, PrivateAttr). This is a **breaking change for any code accessing** `loader.apify_client`.

---

### 14 — 352 unit tests total (322 + 30 new)

**UNVERIFIABLE (precise count)** — pytest is not installed in the environment so the exact parametrized expansion count cannot be confirmed by running the suite. Raw `def test_` function count across all 8 unit test files is 250 (test_actor_tools: 60, test_client: 89, test_deprecated_token_alias: 15, test_document_loaders: 13, test_retrievers: 17, test_tool_response_schema: 3, test_tools: 51, test_clamp_descriptions: 2). Parametrized expansions add at minimum: 30 (clamp) + 9–1 (test_tools) + 6×4+3+5+4+4+4+7+7×3 = ~125 extra = ~375 total, slightly above 352. The 352 figure is approximately plausible but cannot be confirmed precisely without running `pytest --collect-only`. The "30 new in test_clamp_descriptions.py" part is exactly correct.

---

### 15 — README sections for all 3 tool families + retriever + loader

**VALID** — `README.md` contains: `### Core tools` (line 51), `### Search & crawling tools` (line 75), `### Social media tools` (line 99), `## Retriever` (line 165), `## Document loaders` (line 181). All three tool families have dedicated sections with tool-class lists.

---

### 16 — CONTRIBUTING.md is new

**VALID** — Diff shows `new file mode 100644` for `CONTRIBUTING.md`. The file is 65 lines covering contribution flow. `DEVELOPMENT.md` is an *update* (not new) — it had index `2c94ae5..885d8d3`.

---

## Overall Verdict

14 of 16 claims are fully VALID. One claim is STALE (description of which old files were merged), and one is UNVERIFIABLE without a live pytest run. No claims are outright false; the stale claim is a minor misdescription of the file-rename mechanics (single `tools.py` → `tools/actors.py`, not two files collapsed).
