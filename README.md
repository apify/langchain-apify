# langchain-apify

This package enables seamless integration of the Apify platform into your Language Learning Models (LLMs) workflows using LangChain.

## Installation

```bash
pip install -U langchain-apify
```

And you should configure credentials by setting the following environment variables:
- `APIFY_API_TOKEN` - Apify API token

## Tools

`ApifyActorsTool` class exposes Apify actors from Apify. For more information, see [Apify Actors documentation](https://docs.apify.com/platform/actors).

```python
import json
from langchain_apify import ApifyActorsTool

browser = ApifyActorsTool('apify/rag-web-browser')
search_results = browser.invoke(input={
    "run_input": {"query": "what is monero?", "maxResults": 3}
})

# use the tool with an agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
model = ChatOpenAI(model="gpt-4o-mini")
tools = [browser]
agent = create_react_agent(model, tools)

for chunk in agent.stream(
    {"messages": [("human", "search why is monero important?")]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

## Document loaders

`ApifyDatasetLoader` class exposes Apify datasets as document loaders. For more information, see [Apify datasets documentation](https://docs.apify.com/platform/storage/dataset).

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
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"],
        metadata={"source": dataset_item["url"]}
    ),
)
```

## Wrappers

`ApifyWrapper` class wraps the Apify API to easily turn results into documents. For more information, see [Apify LangChain integration documentation](https://docs.apify.com/platform/integrations/langchain).

```python
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

apify = ApifyWrapper()

loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://python.langchain.com/docs/get_started/introduction"}],
        "maxCrawlPages": 10,
        "crawlerType": "cheerio"
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "",
        metadata={"source": item["url"]}
    ),
)
documents = loader.load()
```
