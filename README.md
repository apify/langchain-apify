# langchain-apify

This package enables seamless integration of the Apify platform into your Language Learning Models (LLMs) workflows using LangChain.

## Installation

```bash
pip install -U langchain-apify
```

And you should configure credentials by setting the following environment variables:
- `APIFY_API_TOKEN` - Apify API token

## Tools

`ApifyActorsToolBuilder` class exposes Apify actors from Apify. For more information, see [Apify Actors Documentation](https://docs.apify.com/platform/actors).

```python
import json
from langchain_apify import ApifyActorsTool

browser = ApifyActorsTool(
    #apify_api_token="your-apify-api-token", # Optional, defaults to the APIFY_API_TOKEN environment variable
    actor_id='apify/rag-web-browser'
)
search_results = browser.invoke(input={
    "run_input": {"query": "what is monero?", "maxResults": 3}
})
```

## Document loaders

`ApifyDatasetLoader` class exposes Apify datasets as document loaders. For more information, see [Apify Datasets Documentation](https://docs.apify.com/platform/storage/dataset).

```python
from langchain_apify import ApifyDatasetLoader

# Example dataset structure
# [
#     {
#         "text": "Example text from the website.",
#         "url": "http://example.com"
#     },
#     ...
# ]

loader = ApifyDatasetLoader(
    #apify_api_token="your-apify-api-token", # Optional, defaults to the APIFY_API_TOKEN environment variable
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
    ),
)
```
