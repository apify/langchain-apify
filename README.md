<div align="center">

<picture>
  <img alt="Apify logo" src="https://raw.githubusercontent.com/apify/langchain-apify/refs/heads/main/docs/logo_apify.svg" width="20%" height="20%">
</picture>

LangChain Apify: Full-stack web scraping and data extraction platform enhanced with AI capabilities. Maintained by [Apify](https://apify.com).

<h3>

[Apify](https://apify.com) | [Documentation](https://docs.apify.com/platform/integrations/langchain) | [LangChain](https://langchain.com)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/apify/langchain-apify)](https://github.com/apify/langchain-apify/stargazers)
[![Tests](https://github.com/apify/langchain-apify/actions/workflows/run_code_checks.yml/badge.svg)](https://github.com/apify/langchain-apify/actions/workflows/run_code_checks.yml/badge.svg)

</div>

---

This package allows you to use Apify, a platform for web scraping and data extraction, with LangChain. It provides tools to interact with Apify Actors, datasets, and API.

## Installation

```bash
pip install langchain-apify
```

## Prerequisites

You should configure credentials by setting the following environment variables:
- `APIFY_API_TOKEN` - Apify API token

Register your free Apify account [here](https://console.apify.com/sign-up) and learn how to get your API token in the [Apify documentation](https://docs.apify.com/platform/integrations/api).

## Tools

`ApifyActorsTool` class provides access to [Apify Actors](https://apify.com/store), which are cloud-based web scraping and automation programs that you can run without managing any infrastructure. For more detailed information, see the [Apify Actors documentation](https://docs.apify.com/platform/actors).

`ApifyActorsTool` is useful when you need to run an Apify Actor as a tool in LangChain. You can use the tool to interact with the Actor manually or as part of an agent workflow.

Example usage of `ApifyActorsTool` with the [RAG Web Browser](https://apify.com/apify/rag-web-browser) Actor, which searches for information on the web:
```python
import os
import json
from langchain_apify import ApifyActorsTool

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["APIFY_API_TOKEN"] = "YOUR_APIFY_API_TOKEN"

browser = ApifyActorsTool('apify/rag-web-browser')
search_results = browser.invoke(input={
    "run_input": {"query": "what is Apify Actor?", "maxResults": 3}
})

# use the tool with an agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o-mini")
tools = [browser]
agent = create_react_agent(model, tools)

for chunk in agent.stream(
    {"messages": [("human", "search for what is Apify?")]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

## Document loaders

`ApifyDatasetLoader` class provides access to [Apify datasets](https://docs.apify.com/platform/storage/dataset) as document loaders. Datasets are storage solutions that store results from web scraping, crawling, or data processing.

`ApifyDatasetLoader` is useful when you need to process data from an Apify Actor run. If you are extracting webpage content, you would typically use this loader after running an Apify Actor manually from the [Apify console](https://console.apify.com), where you can access the results stored in the dataset.

Example usage for `ApifyDatasetLoader` with a custom dataset mapping function for loading webpage content and source URLs as a list of  `Document` objects containing the page content and source URL.
```python
import os
from langchain_apify import ApifyDatasetLoader

os.environ["APIFY_API_TOKEN"] = "YOUR_APIFY_API_TOKEN"

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

`ApifyWrapper` class wraps the Apify API to easily convert Apify datasets into documents. It is useful when you need to run an Apify Actor programmatically and process the results in LangChain. Available methods include:

- **call_actor**: Runs an Apify Actor and returns an `ApifyDatasetLoader` for the results.
- **acall_actor**: Asynchronous version of `call_actor`.
- **call_actor_task**: Runs a saved Actor task and returns an `ApifyDatasetLoader` for the results. Actor tasks allow you to create and reuse multiple configurations of a single Actor for different use cases.
- **acall_actor_task**: Asynchronous version of `call_actor_task`.

For more information, see the [Apify LangChain integration documentation](https://docs.apify.com/platform/integrations/langchain).

Example usage for `call_actor` involves running the [Website Content Crawler](https://apify.com/apify/website-content-crawler) Actor, which extracts content from webpages. The wrapper then returns the results as a list of `Document` objects containing the page content and source URL:
```python
import os
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

os.environ["APIFY_API_TOKEN"] = "YOUR_APIFY_API_TOKEN"

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
