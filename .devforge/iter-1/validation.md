# Finding Validation — PR #35 (re-run on opus)

Each panel finding re-checked against the working tree. Verdict per finding:
`CONFIRMED` | `OVERSTATED` (real but impact smaller than claimed) | `FALSE POSITIVE`.

## Confirmed (real, actionable)

### C1 — loader `apify_client → _apify_client` rename, undocumented — CONFIRMED (high)
Diff confirms public field `apify_client: ApifyClient` → `_apify_client: ApifyClient = PrivateAttr()`
(`document_loaders.py:66`). `loader.apify_client` now raises `AttributeError`. `CHANGELOG.md` 0.1.5
section has only one unrelated entry; no `BREAKING CHANGE:` note despite `CONTRIBUTING.md` defining
that convention. Real breaking change with no migration note.

### B2 — `APIFY_API_TOKEN` env var emits no `DeprecationWarning` — CONFIRMED (medium)
`_utils.py:67` `_resolve_apify_token()` returns `os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')`
with no warning. **`DEVELOPMENT.md:55` explicitly states** the env var "emits a `DeprecationWarning`."
Doc-vs-code contradiction. `grep` confirms `test_deprecated_token_alias.py` has zero env-var-path tests.
(README.md:43 and the error string only say "accepted as a deprecated alias" — no warning promise — so
the false claim is isolated to DEVELOPMENT.md.)

### S1 — magic integer literals in `_client.py` — CONFIRMED (major / single-source violation)
`google_maps_search` (`:507` `=10`), `youtube_scrape` (`:729` `=10`), `ecommerce_scrape` (`:771` `=20`)
hardcode defaults. The matching constants **exist and are unused**: `_DEFAULT_GOOGLE_MAPS_MAX_RESULTS=10`,
`_DEFAULT_YOUTUBE_MAX_RESULTS=10`, `_DEFAULT_ECOMMERCE_MAX_RESULTS=20` (`_constants.py:27-29`). `_client.py`
doesn't even import them. Real drift risk against the stated single-source-of-truth design.

### M1 — `tools/__init__.py` `__all__` exports 8 private symbols — CONFIRMED (medium)
`tools/__init__.py:119-127` lists `_DESC_DATASET_ITEMS_LIMIT`, `_DESC_MEMORY_MBYTES`, `_DESC_RUN_TIMEOUT_SECS`,
`_TOOL_RUN_ERRORS`, `_VALID_MEMORY_MBYTES`, `_ApifyGenericTool`, `_iso`, `_run_meta` (8, not 7).
Top-level `__init__.py` is clean; subpackage leaks them via `import *`.

### S2 — `_list_items_or_raise` duplicates `get_dataset_items` — CONFIRMED (medium, quality)
`_client.py:851-857` is identical to `get_dataset_items` (`:139-156`) minus the `offset` param (which
defaults to 0). The two callers pass no offset, so `self.get_dataset_items(id, limit)` is a drop-in.
Real dedup opportunity; not a bug.

### M2 — `ApifyScrapeUrlTool` uses `# noqa: SLF001` to reach `self._client._scrape_url` — CONFIRMED (moderate)
`core.py:331`. Public `scrape_url` (`_client.py:297`) returns only `content`, dropping the
`(run, items, content, source)` tuple the tool needs. A public rich variant would remove the boundary break.

### M4 — `run_actor`/`run_task` share an unabstracted pattern — CONFIRMED (low-moderate, quality)
`_client.py:103-137` and `189-225` are structurally identical bar the SDK method + entity name. Real,
non-blocking.

## Overstated (real but smaller than claimed)

### B1 — `core.py` catches only `RuntimeError`, not `_TOOL_RUN_ERRORS` — OVERSTATED
Confirmed core.py (`:166,216,280,332,392,456`) uses bare `except RuntimeError` and is the **only** module
not importing the shared `_TOOL_RUN_ERRORS = (RuntimeError, ValueError)` that base.py defines and
search.py/social.py both use. **But** the impact claim ("ValueError escapes in production") is wrong for
core tools: the core client methods (`run_actor`, `run_task`, `get_dataset_items`, `_scrape_url`,
`run_*_and_get_items`) raise **only** `RuntimeError`. The `ValueError`-raising methods
(`instagram_scrape`, `twitter_scrape`, `tiktok_scrape`, `youtube_scrape`, `ecommerce_scrape` —
`search_type`/`url_type` validation) are reached only by search/social tools, which DO catch it.
→ Real **consistency / latent-robustness** gap (a future ValueError in a core path would escape),
not an active bug. Still worth fixing for one-line consistency.

### M3 — `ApifySearchRetriever` lacks explicit token-None guard — OVERSTATED (low)
`retrievers.py:81` passes `self.apify_token` straight to `ApifyToolsClient`, skipping the
`if self.apify_token is None: raise` that `base.py:104` performs. Behaviour is correct — the client
constructor handles `None`/`SecretStr` and raises `ValueError` — and the docstring documents the reliance.
Pure consistency nit.

### N4 — 11 tools pass `self.max_timeout_secs` raw instead of clamped — OVERSTATED (nit)
Confirmed (search.py RAG/maps/youtube/ecommerce + all 7 social). But `max_timeout_secs` is the
developer-set ceiling, not LLM input, and these schemas expose no `timeout_secs` field — there is nothing
to clamp. No bug; a documentation/consistency note at most.

## False positives (validation overturned the finding)

### N1 — `docs/examples/tools_example.py` uses `"gpt-5-mini"` — FALSE POSITIVE
`tools_example.py:12` `ChatOpenAI(model='gpt-5-mini')`. The thermonuclear reviewer (sonnet) flagged this
as "non-existent model." **`gpt-5-mini` is a real OpenAI model** (GPT-5 family, released 2025); the
reviewer's training cutoff predates it. The change `gpt-4o-mini → gpt-5-mini` is a valid modernization.
Not a defect.

### N3 — clamp test missing `ApifyRunTaskAndGetDatasetInput.memory_mbytes` — FALSE POSITIVE
The thermonuclear nit claims this tuple is "absent from `_CLAMP_FIELDS`." It is present:
`test_clamp_descriptions.py:50` `(ApifyRunTaskAndGetDatasetInput, 'memory_mbytes', 'max_memory_mbytes')`.
The field IS covered. Reviewer misread the list.

### T1 — `test_field_description_cap_matches_base_class_default` is a tautology — FALSE POSITIVE (overturned)
code-review (FAIL) vs staff-review (PASS) disagreed; validation sides with staff-review. The test reads the
**description string** off the input-schema Field, regex-extracts the advertised number, and compares it to
the base-class `max_*` default. If someone hardcodes a description to `"clamped to 300 max"` while the
constant/default stays 600, `advertised=300 != actual=600` → the test **fails and catches it**. code-review's
reasoning ("only detects when both are wrong by the same amount") is incorrect. The only thing it can't
"catch" is an intentional constant change — which correctly updates both sides. The test is a valid drift guard.

## Net result

- 7 confirmed (1 high, 1 medium-bug, 1 major-quality, 3 medium, plus M4 low) — all real.
- 3 overstated (B1, M3, N4) — real but lower severity / not active bugs.
- 3 false positives (N1, N3, T1) — drop.

The two genuine merge-blockers that survive validation: **C1** (undocumented breaking change) and
**B2** (doc promises a deprecation warning the code never emits). Everything else is quality/consistency.
