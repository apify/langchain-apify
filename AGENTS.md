# Coding guidelines

This file provides guidance to programming agents when working with code in this repository.

[LangChain](https://www.langchain.com/) integration for the [Apify](https://apify.com/) platform, published on PyPI as `langchain-apify`. The public API is what `src/langchain_apify/__init__.py` exports. The generic tools live in `tools/core.py`, `tools/search.py` and `tools/social.py` on the `_ApifyGenericTool` base in `tools/base.py`, the legacy `ApifyActorsTool` in `tools/actors.py`, `ApifyWrapper` in `wrappers.py`, the loaders in `document_loaders.py`, and `ApifySearchRetriever` in `retrievers.py`. Modules whose names start with `_` are private.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for issue, commit and PR conventions, and [DEVELOPMENT.md](DEVELOPMENT.md) for setup, tasks and tests. Both apply to agents too.

@CONTRIBUTING.md
@DEVELOPMENT.md

## Components

- Generic tools subclass `_ApifyGenericTool` and declare only `name`, `description`, `args_schema` and `_client_call`, which returns the `ApifyToolsClient` method to call and its arguments. The base runs that method in `_run` and awaits its `a`-prefixed async version in `_arun`, so every client method a tool uses needs both. Tools return the JSON envelope `{"run": {...}, "items": [...]}` and pass values the LLM supplies through the `_clamp_*` helpers on the base.
- In `_client.py`, each domain method builds its Actor call in a `*_call` method, and `run_call` or `arun_call` runs it, so the sync and async paths send the same input.
- Settings the LLM must not change, like `max_total_charge_usd` and `mark_output_untrusted`, are tool fields or constructor arguments, never tool arguments.
- `ApifyActorsTool` reads the Actor's latest build when it is constructed and builds its argument schema from the Actor's input schema. The build is a pydantic model from apify-client dumped to a dict, so a field the API leaves out is present as `None`, not missing.
- Every Apify client comes from `_create_apify_client` in `_utils.py`, which adds the `x-apify-integration-platform: langchain` header and the `Origin/langchain` user-agent marker.
- The token is `apify_token`, a `SecretStr` that falls back to the `APIFY_TOKEN` environment variable. `apify_api_token` and `APIFY_API_TOKEN` are deprecated aliases handled in `_utils.py`; don't add new fields for them.
- Shared defaults live in `_constants.py`. The package imports from `langchain_core` only, never from `langchain`, `langchain_community` or `langchain_experimental` (see `scripts/lint_imports.sh`).
- Unit tests run with network access disabled by `pytest-socket` and never call the Apify API. Generic tools are built with `make_tool` from `tests/unit_tests/conftest.py`, which swaps in a mocked `ApifyToolsClient`.
- Integration tests in `tests/integration_tests` call the real Apify API with the `APIFY_TOKEN` token and skip without it. Keep Actor runs few and small.

## Conventions

- Run `uv run poe check-code` before finishing a change.
- New modules start with `from __future__ import annotations`. Use single quotes, and follow the Google docstring convention.
- Commits and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/), with the types listed in CONTRIBUTING.md.
