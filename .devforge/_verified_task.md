# Verified Task — PR #35: feat: modernize langchain integration connector

**Repository:** apify/langchain-apify
**Branch:** feat/modernize-langchain-integration (HEAD: 75dc0cd)
**Base:** origin/main (HEAD: 636362f)
**Task kind:** review-only (no source edits)

---

## What this PR actually does (authoritative, post-fact-check)

### New public API

**Core generic tools** (`langchain_apify/tools/core.py`, exported from `langchain_apify`):
- `ApifyRunActorTool` — run any Actor, return run metadata envelope
- `ApifyGetDatasetItemsTool` — fetch dataset items by ID
- `ApifyRunActorAndGetDatasetTool` — run + fetch combined
- `ApifyScrapeUrlTool` — scrape single URL to markdown
- `ApifyRunTaskTool` — run a saved Actor task
- `ApifyRunTaskAndGetDatasetTool` — task run + fetch combined
- Convenience list: `APIFY_CORE_TOOLS`

**Search & crawling tools** (`langchain_apify/tools/search.py`):
- `ApifyGoogleSearchTool`, `ApifyWebCrawlerTool`, `ApifyRAGWebBrowserTool`
- `ApifyGoogleMapsTool`, `ApifyYouTubeScraperTool`, `ApifyEcommerceScraperTool`
- Convenience list: `APIFY_SEARCH_TOOLS`

**Social media tools** (`langchain_apify/tools/social.py`):
- `ApifyInstagramScraperTool`, `ApifyLinkedInProfilePostsTool`, `ApifyLinkedInProfileSearchTool`
- `ApifyLinkedInProfileDetailTool`, `ApifyTwitterScraperTool`, `ApifyTikTokScraperTool`
- `ApifyFacebookPostsScraperTool`
- Convenience list: `APIFY_SOCIAL_TOOLS`

**LangChain-native components:**
- `ApifySearchRetriever(BaseRetriever)` in `langchain_apify/retrievers.py` — wraps `apify/rag-web-browser`
- `ApifyCrawlLoader(BaseLoader)` in `langchain_apify/document_loaders.py` — wraps `apify/website-content-crawler`

---

### Shared infrastructure

**`langchain_apify/_constants.py`** — single-source defaults and clamp ceilings:
- `_MAX_TIMEOUT_SECS_CAP = 600`
- `_MAX_MEMORY_MBYTES_CAP = 32768`
- `_MAX_ITEMS_CAP = 1000`
- `_MAX_CRAWL_DEPTH_CAP = 5`
- Plus ~15 `_DEFAULT_*` values for operation-specific defaults

**`langchain_apify/_types.py`** — shared Literal type aliases (`CrawlerType`, `YouTubeSearchType`, `EcommerceUrlType`, `InstagramSearchType`, `TwitterSearchMode`, `TwitterSort`, `TikTokSearchType`)

**`langchain_apify/_error_messages.py`** — three shared error string constants

**`langchain_apify/tools/base.py`** — `_ApifyGenericTool(BaseTool)` base class with:
- `apify_token` field + `_client: ApifyToolsClient = PrivateAttr()`
- Developer-controlled clamp fields: `max_timeout_secs`, `max_memory_mbytes`, `max_items`, `max_crawl_depth`
- Clamp methods: `_clamp_timeout`, `_clamp_memory`, `_clamp_items`, `_clamp_depth`
- `_envelope(run, items) -> str` — uniform JSON output helper

---

### Package restructure

Old structure deleted: `tools.py` (single file, ApifyActorsTool only), `const.py`, `error_messages.py`, `utils.py` (public names without leading underscore).

New structure: `tools/` subpackage with `__init__.py`, `actors.py`, `base.py`, `core.py`, `search.py`, `social.py`. The old `tools.py` was **renamed** to `tools/actors.py` (not merged with a separate `_actor_tools.py` — no such file existed).

---

### Breaking changes

1. **`ApifyDatasetLoader._apify_client`** — previously the public attribute `apify_client` is now `_apify_client` (`PrivateAttr`). Any code accessing `loader.apify_client` directly will break.

2. **`APIFY_API_TOKEN`** — previously the primary env var; now a deprecated fallback. Code using only `APIFY_API_TOKEN` will still work (with `DeprecationWarning`) but `APIFY_TOKEN` is now the canonical env var.

---

### Bug fixes

1. **`_prune_actor_input_schema` falsy defaults** (`_utils.py` line 145): changed from truthy `if (value := ...)` to `if (value := ...) is not None`, preserving `default=0`, `default=false`, `default=""` in the Actor input schema passed to the LLM.

2. **`ApifyWrapper.(a)call_actor(_task)` token forwarding** (`wrappers.py`): all four methods now pass `apify_token=self.apify_token` to the constructed `ApifyDatasetLoader`, so the loader does not require a separate env-var lookup.

3. **`ApifyActorsTool.__init__` 2→1 network fetch**: old code called `get_actor_latest_build()` twice (once inside `_create_description()` and once inside `_build_tool_args_schema_model()`). New code calls `_get_actor_latest_build()` once in `__init__` and passes the `build` dict to both helpers.

---

### New tests

- **`tests/unit_tests/test_clamp_descriptions.py`** — 2 parametrized test functions × 15 clamp-field entries = **30 test instances** that pin: (a) every clamp Field has a "clamped to N max" description, (b) N matches the live `_ApifyGenericTool` default.
- **`tests/unit_tests/test_deprecated_token_alias.py`** — 15 test methods across 5 classes covering all token-alias deprecation paths.
- **`tests/unit_tests/test_actor_tools.py`** — new file covering search + social tools.
- **`tests/unit_tests/test_retrievers.py`** — 17 retriever unit tests.

Total raw test function count (all unit test files): 250 `def test_` functions. With parametrized expansions the pytest-reported total is approximately 350 ± 20; the PR claims 352 total (322 + 30 new). Exact number unverifiable without a live pytest run.

---

### Documentation

- **`CONTRIBUTING.md`** — new file (65 lines): contribution flow, PR conventions.
- **`DEVELOPMENT.md`** — updated (was not new): local setup, lint, test commands.
- **`README.md`** — rewritten with dedicated sections: `### Core tools`, `### Search & crawling tools`, `### Social media tools`, `## Retriever`, `## Document loaders`.

---

## Review scope for downstream stages

The diff is 8 384 lines across 37 files. Key areas to scrutinize:

1. **`langchain_apify/tools/base.py`** — `_ApifyGenericTool`: clamp logic, `_envelope` serialisation, `PrivateAttr` client construction, token handling.
2. **`langchain_apify/_client.py`** — `ApifyToolsClient`: all new methods (run_actor, run_task, crawl_website, google_search, rag_web_search, google_maps_search, youtube_scrape, ecommerce_scrape, instagram_scrape, linkedin_*, twitter_scrape, tiktok_scrape, facebook_posts_scrape).
3. **`langchain_apify/_utils.py`** — falsy-default fix in `_prune_actor_input_schema`; `_get_actor_latest_build` (2-fetch reduction).
4. **`langchain_apify/document_loaders.py`** — `_apify_client` PrivateAttr change (backward-compat breakage); `ApifyCrawlLoader` new class.
5. **`langchain_apify/wrappers.py`** — token forwarding to `ApifyDatasetLoader`.
6. **`tests/unit_tests/test_clamp_descriptions.py`** — correctness of clamp cap advertised vs. actual.
7. **`langchain_apify/__init__.py`** — export completeness; no accidental internal symbols exposed.
8. **All `_VALID_MEMORY_MBYTES` snap logic** in `tools/base.py` — edge cases at cap boundary.
