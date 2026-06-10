_ERROR_APIFY_TOKEN_ENV_VAR_NOT_SET = (
    'APIFY_TOKEN environment variable is not set.'
    ' Please set it to your Apify API token by using `os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"`'
    ' in your code or pass it as environment variable.'
    ' To pass it as environment variable, you can use the following command:'
    ' `APIFY_TOKEN="YOUR_APIFY_TOKEN" python your_script.py`'
)

_ERROR_ACTOR_RUN_FAILED = 'Actor run {run_id} ended with status {status}.'

_ERROR_SCRAPE_EMPTY = 'No content extracted from {url}.'
