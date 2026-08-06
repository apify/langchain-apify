# Development

This file covers everything you need to run the code locally: install, format, lint, and test.

> Planning a pull request? Read [CONTRIBUTING.md](CONTRIBUTING.md) for issue and PR conventions before opening one.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [Hatch](https://hatch.pypa.io/) as the build backend and task runner.

Clone the repository and install the project with its development dependencies:

```bash
git clone https://github.com/apify/langchain-apify
cd langchain-apify

uv sync --extra dev
```

Development tasks are defined as Hatch scripts in `pyproject.toml` and run with `hatch run <task>`. Hatch manages an isolated environment with the dev dependencies, so install it once (e.g. `uv tool install hatch` or `pipx install hatch`).

## Formatting and linting

To format the code, use the following command:

```bash
hatch run format
```

To lint and type-check the code, use the following commands:

```bash
hatch run lint
hatch run typecheck
```

## Testing

To run unit tests, use the following command:

```bash
hatch run test
```

To run integration tests, use the following command:

```bash
APIFY_TOKEN="YOUR_TOKEN" hatch run integration-test
```

To run a single test file, pass it as an argument:

```bash
hatch run test tests/unit_tests/test_file.py
APIFY_TOKEN="YOUR_TOKEN" hatch run integration-test tests/integration_tests/test_file.py
```

> `APIFY_API_TOKEN` is also accepted as a deprecated alias for `APIFY_TOKEN` (emits a `DeprecationWarning`). New code and examples should use `APIFY_TOKEN`.
