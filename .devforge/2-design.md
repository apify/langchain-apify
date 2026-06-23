# Design — PR #35: feat: modernize langchain integration connector

**Review-only run. No source edits.**

---

## What we're checking

- **Correctness of new shared infrastructure**: clamp logic, _envelope serialisation, and PrivateAttr
  client construction in `tools/base.py` are the load-bearing foundation every tool subclass inherits.
  Errors here propagate to all 19 tool classes.
- **Public API contract**: the package restructure (`tools/` subpackage) must preserve every symbol
  that existed in the old flat layout; `__init__.py` export completeness is the single gate.
- **Breaking changes land safely**: the `apify_client → _apify_client` rename and the
  `APIFY_TOKEN / APIFY_API_TOKEN` swap are known backward-compat breaks with no migration guide in
  the PR body. Review must assess severity and whether a deprecation shim or CHANGELOG entry is
  sufficient.
- **Bug-fix correctness**: three targeted fixes (falsy-default schema pruning, token forwarding in
  wrappers, 2→1 network fetch) are individually simple but interact with downstream code; verify
  the fix is complete and has no off-by-one or None-edge cases.
- **Test suite integrity**: 30 new parametrized clamp tests are the primary regression harness for
  the infrastructure layer; their invariants (advertised cap == live default) must actually fire on
  a live class instance, not a stale constant.

---

## Review focus areas

### 1. `langchain_apify/tools/base.py` — _ApifyGenericTool

Invariants to check:
- Clamp methods must be monotone: any input ≤ cap returns unchanged; any input > cap returns exactly
  the cap value; `None` input returns the field default (not `None`).
- `_envelope(run, items)` must produce valid JSON under all item types (list, empty list, None);
  check that the run-metadata dict is never mutated before serialisation.
- `PrivateAttr` client: verify `model_post_init` (or `__init__`) constructs `_client` from
  `apify_token` and that a missing token raises the correct error, not an `AttributeError` on first
  use.
- Token resolution order: explicit `apify_token` field → `APIFY_TOKEN` env var → `APIFY_API_TOKEN`
  env var (deprecated). Confirm no path silently falls back to `None`.

### 2. `langchain_apify/_client.py` — ApifyToolsClient

Invariants to check:
- Each of the 23+ platform methods maps to exactly one Actor ID (or task ID). Confirm no method
  shares an ID with another, and that IDs match the official Apify Store slugs (typo risk in
  string literals).
- Input dicts passed to `run_actor` / `run_task`: verify required fields are not accidentally
  dropped when optional params are `None`. Check that `None` values are either excluded from the
  payload or are accepted by the target Actor.
- Error propagation: if a run fails (non-SUCCEEDED status), does the client raise or silently return
  empty results?

### 3. `langchain_apify/_utils.py` — falsy-default fix + _get_actor_latest_build

Invariants to check:
- `_prune_actor_input_schema`: with the `is not None` fix, confirm that `default=0`, `default=False`,
  and `default=""` are all preserved; also confirm `default=None` (explicit null) is still pruned.
- `_get_actor_latest_build`: single-call path must handle "no build found" (Actor with no successful
  build) without raising; the result is passed downstream to `_create_description` and
  `_build_tool_args_schema_model` — both callers must tolerate `None` or empty build gracefully.

### 4. `langchain_apify/document_loaders.py` — breaking rename + ApifyCrawlLoader

Invariants to check:
- `_apify_client` as `PrivateAttr`: confirm the attribute is initialised before `lazy_load` is
  called; a partial-init loader must not silently return an empty document list.
- `ApifyCrawlLoader`: verify it is exported from `__init__.py` and that its LangChain `BaseLoader`
  contract (`lazy_load` yields `Document` objects with non-empty `page_content`) is correctly
  implemented.
- Breaking change severity: `loader.apify_client` access will raise `AttributeError` with no
  warning. Check whether the PR includes a CHANGELOG entry or migration note for downstream
  consumers.

### 5. `langchain_apify/__init__.py` — export completeness

Invariants to check:
- Every class exported from the old single-file layout (`ApifyDatasetLoader`, `ApifyWrapper`,
  `ApifyActorsTool`) must still be importable at `langchain_apify.<Name>`.
- New exports (`APIFY_CORE_TOOLS`, `APIFY_SEARCH_TOOLS`, `APIFY_SOCIAL_TOOLS`, all 19 tool classes,
  `ApifySearchRetriever`, `ApifyCrawlLoader`) must be present.
- No internal symbols (leading-underscore classes, `_ApifyGenericTool`, `_constants`, `_types`)
  should appear in `__all__` or be reachable via a bare `from langchain_apify import *`.

### 6. `tests/unit_tests/test_clamp_descriptions.py` — drift protection

Invariants to check:
- `test_field_description_cap_matches_base_class_default`: this test reads the live `Field` default
  from `_ApifyGenericTool` at test time; confirm it does not import the constant directly (which
  would make it trivially pass even if the class field defaulted to a different value).
- Parametrize coverage: 15 entries must cover all four cap constants
  (`_MAX_TIMEOUT_SECS_CAP`, `_MAX_MEMORY_MBYTES_CAP`, `_MAX_ITEMS_CAP`, `_MAX_CRAWL_DEPTH_CAP`)
  as well as the operation-specific defaults. Confirm no cap constant is missing from the list.

---

## Risks flagged by fact-check

| Risk | Severity | Status |
|------|----------|--------|
| `apify_client → _apify_client` rename: silent `AttributeError` for all existing consumers of `ApifyDatasetLoader` | High | No migration guide or deprecation shim in PR |
| `APIFY_API_TOKEN` demoted: existing deployments using only this env var get a `DeprecationWarning`; env var still works | Medium | Acceptable, but warning text should name the replacement clearly |
| PR description says "collapsed `_actor_tools.py` + `tools.py`" — `_actor_tools.py` never existed; actual change was a rename | Low | Cosmetic inaccuracy; corrected in `_verified_task.md` |
| Test count (352) is approximately plausible but unverifiable without a live pytest run | Low | Reviewers should note if they observe a discrepancy during local runs |
| 8 384-line diff across 37 files with no sub-PR breakdown in review tooling | Medium | Review panel must be thorough; no single reviewer should sign off alone |

---

## Panel

**Complexity**: large (37 files, 8 384 lines, breaking changes, new public API surface).

| Role | Reviewer | Responsibility |
|------|----------|---------------|
| Lead reviewer | `staff-review` | Full diff pass; public API contract; breaking change assessment; CHANGELOG/migration check |
| Infrastructure reviewer | `thermonuclear` | Adversarial deep-dive on `tools/base.py`, `_client.py`, `_utils.py`; edge cases; error paths |
| Code quality reviewer | `code-review` | Consistency, style, correctness of test suite; export hygiene in `__init__.py` |

**Final sign-off required from**: `thermonuclear` + `code-review` (both must approve before merge).

Staff-review may approve independently; merge requires all three to have completed their pass.
