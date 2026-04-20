ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET = (
    'APIFY_API_TOKEN environment variable is not set.'
    ' Please set it to your Apify API token by using `os.environ["APIFY_API_TOKEN"] = "YOUR_APIFY_API_TOKEN"'
    ' in your code or pass it as environment variable.'
    ' To pass it as environment variable, you can use the following command:'
    ' `APIFY_API_TOKEN="YOUR_APIFY_API_TOKEN" python your_script.py`'
)

ERROR_ACTOR_RUN_FAILED = 'Actor run {run_id} ended with status {status}.'

ERROR_SCRAPE_EMPTY = 'No content extracted from {url}.'
