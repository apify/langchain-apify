# Development

## Contributing

If you want to contribute, please ensure that you have your environment properly set up. Run the following commands to make sure that the code is properly formatted and is not breaking before submitting a pull request.

## Installation

To work on this repo locally, you first need to clone the repository and install the dependencies. You can do this by running the following commands:

```bash
git clone https://github.com/apify/langchain-apify
cd langchain-apify

poetry sync --all-groups
# for poetry version < 2.0
poetry install --with dev,test,lint --no-root --sync
```

## Formatting and linting

To format the code, use the following command:

```bash
make format
```

To lint the code, use the following command:

```bash
make lint
```

## Testing

To run unit tests, use the following command:

```bash
make test
```

To run integration tests, use the following command:

```bash
APIFY_API_TOKEN="YOUR_TOKEN" make integration_test
```

To run single test file, use `TEST_FILE` argument:

```bash
make test TEST_FILE=path_to/test_file.py
APIFY_API_TOKEN="YOUR_TOKEN" make integration_test TEST_FILE=path_to/test_file.py
```
