This is a development documentation for this project.

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
make integration_test TEST_FILE=path_to/test_file.py
```
