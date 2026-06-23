VERDICT: FAIL

---

## Findings

### [major] Magic literals in `_client.py` for `google_maps_search`, `youtube_scrape`, `ecommerce_scrape` (lines 507, 729, 771)

Three methods in `_client.py` use raw integer default arguments (`max_results: int = 10` for `google_maps_search` and `youtube_scrape`, `max_results: int = 20` for `ecommerce_scrape`) instead of the named constants that exist in `_constants.py` (`_DEFAULT_GOOGLE_MAPS_MAX_RESULTS = 10`, `_DEFAULT_YOUTUBE_MAX_RESULTS = 10`, `_DEFAULT_ECOMMERCE_MAX_RESULTS = 20`). The corresponding tool-layer schemas in `search.py` correctly reference the constants, so the defaults are numerically consistent today—but the client-layer defaults are now decoupled from the constants and will silently diverge if a constant changes. The entire point of `_constants.py` is to be the single source of truth; these three omissions undermine that guarantee.

**Files:** `langchain_apify/_client.py` lines 507, 729, 771.

---

### [major] `_list_items_or_raise` is a near-duplicate of `get_dataset_items` in `_client.py`

`_list_items_or_raise(dataset_id, limit)` (line 851) and `get_dataset_items(dataset_id, limit, offset=0)` (line 139) are structurally identical except that the public method also accepts `offset` and passes it to the SDK. `_list_items_or_raise` could be replaced by `self.get_dataset_items(dataset_id, limit)` throughout—`run_actor_and_get_items` and `run_task_and_get_items` both call only `_list_items_or_raise`, so they would work identically. The private helper adds no safety that isn't already in the public one; it just creates another divergence surface (the public method was updated to add `offset` support, the private one was not). This is a quiet spaghetti seam inside a class the design doc already flagged as prone to god-class growth.

**Files:** `langchain_apify/_client.py` lines 139–156 and 851–857.

---

### [moderate] `ApifyScrapeUrlTool._run` accesses a private method of a collaborator via `# noqa: SLF001`

`core.py` line 331:
```python
run, _, content, _ = self._client._scrape_url(url, self._clamp_timeout(timeout_secs))  # noqa: SLF001
```

The comment says `scrape_url()` drops the metadata this tool needs. The clean fix is to make `_scrape_url` a proper public method (rename → `scrape_url_full` or `scrape_url_with_run`) and let `scrape_url` be the thin public alias—or to add a return-value variant to the public interface. Silencing SLF001 normalises reaching across a collaboration boundary in a way that will be invisible to future callers who try to build on `ApifyToolsClient`. This is exactly the kind of quiet coupling the design doc asks thermonuclear to flag.

**File:** `langchain_apify/tools/core.py` line 331; `langchain_apify/_client.py` lines 297–303.

---

### [moderate] `ApifySearchRetriever.model_post_init` does not validate `apify_token is None` before constructing `ApifyToolsClient`

`_ApifyGenericTool.model_post_init` (base.py line 104) explicitly checks `if self.apify_token is None: raise ValueError(...)` before creating the client. `ApifySearchRetriever.model_post_init` (retrievers.py lines 78–82) skips this check and passes `apify_token=self.apify_token` directly to `ApifyToolsClient.__init__`. The client constructor handles the `None` path by calling `_resolve_apify_token()` again, so the behaviour is accidentally correct—but the contract is inconsistent: the retriever relies on the client to do the token check, while the tool base class does it locally. If the client's fallback logic ever changes, the retriever silently loses its guard. The check should be explicit in `model_post_init`, matching the pattern in `_ApifyGenericTool`.

**File:** `langchain_apify/retrievers.py` lines 78–82.

---

### [moderate] `run_actor` and `run_task` share a structural pattern that is not abstracted

Both `run_actor` (lines 103–137) and `run_task` (lines 189–225) follow an identical call–check–return pattern:

1. Build `call_kwargs` with conditional `memory_mbytes`.
2. Call the SDK endpoint (`.actor().call()` vs `.task().call()`), catch `_TRANSPORT_EXCEPTIONS`.
3. Check `run is None`.
4. Call `_check_run_status(run)`.
5. Return `run`.

The only variation is the SDK client method called and the entity name in error messages. `run_actor_and_get_items` / `run_task_and_get_items` duplicate this same step-4–5 and then add a dataset fetch. A thin private helper `_run_and_check(callable, entity_name, call_kwargs) -> dict` would collapse both pairs. The duplication is not accidental; it tracks a structural smell in a class the design doc flagged as god-class territory.

**File:** `langchain_apify/_client.py` lines 103–137 and 189–225.

---

### [minor] `_prune_actor_input_schema` treats `prefill` identically to `default`, but `prefill` is not a standard JSON Schema key

`_utils.py` line 144 iterates `('type', 'default', 'prefill', 'enum')` and emits all four verbatim into the pruned schema. `prefill` is an Apify-specific hint that LLMs and JSON-Schema validators won't understand. Emitting it is probably harmless, but silently forwarding it without documentation makes the schema opaque. At minimum, a comment explaining the intent would prevent future removal by a maintainer who doesn't know the Apify actor schema contract.

**File:** `langchain_apify/_utils.py` line 144.

---

### [minor] `docs/examples/tools_example.py` references a non-existent model `"gpt-5-mini"`

The diff changes `model='gpt-4o-mini'` → `model='gpt-5-mini'` (line 403 in the diff). No such OpenAI model exists as of this PR. This is clearly a typo introduced during the update. The example is broken.

**File:** `docs/examples/tools_example.py` line 7 (post-patch).

---

### [minor] `_clamp_memory` edge case: `max_memory_mbytes` set below 128 returns `_VALID_MEMORY_MBYTES[0]` (128 MB), ignoring the configured cap

`_clamp_memory` guards against an out-of-bounds index with `return _VALID_MEMORY_MBYTES[max(idx, 0)]`. If a developer misconfigures `max_memory_mbytes=64` (below the platform minimum), the method returns 128 MB—silently overriding the developer's cap. The inline comment says "Misconfigured cap below the platform minimum, return the minimum." but there is no warning, no documented exception, and no test for this path. The _ceiling intent_ of the cap (prevent the LLM from requesting more) is preserved, but the _value intent_ (developer sets 64 as their cap) is silently violated. A `warnings.warn` here would align with the deprecation warnings used elsewhere in the package.

**File:** `langchain_apify/tools/base.py` lines 119–122.

---

### [nit] `test_clamp_descriptions.py` covers 15 entries but misses `ApifyRunActorInput.memory_mbytes` at item level

The `_CLAMP_FIELDS` list (line 36) correctly covers `memory_mbytes` for `ApifyRunActorInput` and `ApifyRunTaskInput`, but the test fixture comment says "15 entries" against "4 cap constants + operation-specific defaults." All four caps are covered for at least one schema, but `memory_mbytes` is not tested for `ApifyRunTaskAndGetDatasetInput`—that schema has it (line 103 of core.py) but the tuple is absent from `_CLAMP_FIELDS`. This means the `max_memory_mbytes` drift test has a gap for the task-and-get-dataset combined tool.

**File:** `tests/unit_tests/test_clamp_descriptions.py` lines 36–52.

---

## Summary

Two majors prevent merge: (1) magic literals in `_client.py` for three methods that bypass the single-source constants, and (2) the `_list_items_or_raise` / `get_dataset_items` near-duplicate that creates a silent divergence seam. The moderate findings (SLF001 coupling, missing retriever token guard, duplicated run/task control flow) are not blockers on their own but collectively indicate that the god-class concern raised in `2-design.md` was not fully addressed—the client class has grown to 23+ methods with duplicated patterns that a light internal refactor could eliminate. The typo in the example file is a blocking nit for a public-facing doc.
