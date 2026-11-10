# Contributing

Thanks for considering a contribution to `langchain-apify`. This file covers the contribution flow: what to file, how to scope a PR, and what review looks for.

> Setting up locally for the first time? Start with [DEVELOPMENT.md](DEVELOPMENT.md). Install, lint, and test commands live there.

## Filing issues

Open a GitHub issue at <https://github.com/apify/langchain-apify/issues>. A few notes:

- **Bug reports**: include the package version, a short reproducer, the expected behaviour, and what you saw instead. If you have an Apify run ID from the failing call, paste it; it's the fastest way for us to inspect the run server-side.
- **Feature requests**: describe the use case before the proposed API. We'd rather discuss the shape of the solution before code lands.
- Search existing issues first; the maintainers may already be tracking the same thing.

## Working on a pull request

- Branch from `main`.
- Keep one logical change per PR. A PR that fixes a bug *and* adds a feature is harder to review and harder to revert if needed.
- Before opening: run `uv run poe check-code` locally. CI will also run integration tests; you don't need an `APIFY_TOKEN` to open the PR (CI has its own).
- If your change affects the public API, update the README and any in-repo examples.
- If your change adds a new tool family or generic primitive, add or extend the corresponding test file under `tests/unit_tests/`.

## Commit message conventions

The release workflows use `git-cliff` to read commit-message prefixes and auto-generate both the version bump and the changelog. **Don't manually edit `version =` in `pyproject.toml` or write `CHANGELOG.md` entries by hand.**

Use these [Conventional Commits](https://www.conventionalcommits.org/) types:

- `feat:` for a new feature
- `fix:` for a bug fix
- `perf:` for a performance improvement
- `refactor:` or `style:` for a source change that doesn't change behaviour
- `test:`, `docs:`, `build:`, `chore:` or `ci:` for everything else

A `feat`, `fix`, `perf`, `refactor` or `style` commit on `main` publishes a beta, so use those types for source changes only. The changelog lists `feat`, `fix`, `perf` and `revert` commits, plus breaking changes of any type.

Mark a breaking change with `!` after the type (`feat!:`) or a `BREAKING CHANGE:` footer. Use it sparingly and only when the public API genuinely breaks. While the version is below 1.0, a breaking change bumps the minor version and a feature bumps the patch version.

A good message is short on the subject line and explains *why* (not what) in the body:

```
fix: forward apify_token from ApifyWrapper to ApifyDatasetLoader

Without this, an explicit wrapper token still required APIFY_TOKEN to
also be set in the environment. The loader fell back to env-var
resolution and raised ValueError if neither was present.
```

## What review looks for

- **Correctness on the public API surface.** Any new tool must follow the `_ApifyGenericTool` envelope contract (a JSON string of `{"run": {...}, "items": [...]}`) and route Actor calls through `ApifyToolsClient` (`_client.py`), not the SDK directly.
- **`uv run poe check-code` passes locally.** Integration tests pass under CI's token; you don't need to run them yourself unless you're touching `_client.py`.
- **No new `apify_api_token` field declarations.** The canonical token kwarg is `apify_token`; the legacy `apify_api_token` is honoured only via the existing deprecation plumbing in `_utils.py` and per-tool model validators. New code should not introduce fresh `apify_api_token` fields.
- **No manually bumped `version =` in `pyproject.toml`.** Versions come from commit messages via `git-cliff`.
- **Shared defaults stay in `_constants.py`.** Don't reintroduce magic literals (`300`, `100`, `120`, etc.); import the named constant instead.

## Releases

- **Betas are automatic.** After a PR merges to `main`, `on_main.yaml` runs the checks and, for a `feat`, `fix`, `perf`, `refactor` or `style` commit, dispatches `manual_release_beta.yaml`, which publishes a beta to PyPI.
- **Stable releases are manual.** A maintainer runs `manual_release_stable.yaml` from `main`. With `auto`, `git-cliff` picks the version from the commits since the last tag; `custom` takes an exact version. The workflow bumps `pyproject.toml`, writes the changelog, tags the release, creates the GitHub release and publishes to PyPI.

Neither needs any manual version or changelog edits.
