# Development

This file covers everything you need to run the code locally: install, format, lint, and test.

> Planning a pull request? Read [CONTRIBUTING.md](CONTRIBUTING.md) for issue and PR conventions before opening one.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [Poe the Poet](https://poethepoet.natn.io/) as the task runner.

Clone the repository and install the project with its development dependencies:

```bash
git clone https://github.com/apify/langchain-apify
cd langchain-apify

uv run poe install-dev
```

Development tasks are defined as Poe tasks in `pyproject.toml` and run with `uv run poe <task>`. Run `uv run poe` to list them.

## Formatting and linting

To format the code, use the following command:

```bash
uv run poe format
```

To lint and type-check the code, use the following commands:

```bash
uv run poe lint
uv run poe type-check
```

To run all checks and the unit tests before opening a pull request:

```bash
uv run poe check-code
```

## Testing

To run unit tests, use the following command:

```bash
uv run poe unit-tests
```

To run integration tests, use the following command:

```bash
APIFY_TOKEN="YOUR_TOKEN" uv run poe integration-tests
```

To run a single test file, call pytest directly:

```bash
uv run pytest --disable-socket --allow-unix-socket tests/unit_tests/test_file.py
APIFY_TOKEN="YOUR_TOKEN" uv run pytest tests/integration_tests/test_file.py
```

> `APIFY_API_TOKEN` is also accepted as a deprecated alias for `APIFY_TOKEN` (emits a `DeprecationWarning`). New code and examples should use `APIFY_TOKEN`.
