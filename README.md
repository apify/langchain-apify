> 🎉 **Apify MCP server released!** 🎉
>
> Apify has released its MCP ([Model Context Protocol](https://modelcontextprotocol.io)) server, which offers more features. You can use it through the [LangChain MCP Adapter](https://github.com/langchain-ai/langchain-mcp-adapters). It allows you to run Apify Actors, access Apify storage, search and read Apify documentation, and much more.
>
> ### 👉 [https://mcp.apify.com](https://mcp.apify.com) 👈

<div align="center">

<picture>
  <img alt="Apify logo" src="https://raw.githubusercontent.com/apify/langchain-apify/refs/heads/main/docs/apify-logo.png" width="20%" height="20%">
</picture>

LangChain Apify: A full-stack scraping platform built on Apify's infrastructure and LangChain's AI tools. Maintained by [Apify](https://apify.com).

<h3>

[Apify](https://apify.com) | [Documentation](https://docs.apify.com/platform/integrations/langchain) | [LangChain](https://langchain.com)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/apify/langchain-apify)](https://github.com/apify/langchain-apify/stargazers)
[![Tests](https://github.com/apify/langchain-apify/actions/workflows/run_code_checks.yml/badge.svg)](https://github.com/apify/langchain-apify/actions/workflows/run_code_checks.yml/badge.svg)

</div>

---

Build web scraping and automation workflows in Python by connecting Apify Actors with LangChain. This package gives you programmatic access to Apify's infrastructure: run scraping tasks, handle datasets, and use the API directly through LangChain's tools.

## Agentic LLMs

If you are an agent or an LLM, refer to the [llms.txt](llms.txt) file to get package context and learn how to work with this package.

## Installation

```bash
pip install langchain-apify
```

## Prerequisites

You should configure credentials by setting the following environment variable:
- `APIFY_TOKEN`: Apify API token. (`APIFY_API_TOKEN` is also honoured as a deprecated alias for backwards compatibility.)

Register your free Apify account [here](https://console.apify.com/sign-up) and learn how to get your API token in the [Apify documentation](https://docs.apify.com/platform/integrations/api).

## Tools

The package ships dedicated tools across three families plus a generic "wrap any Actor by ID" tool for everything else. All return a uniform `{"run": {...}, "items": [...]}` JSON envelope (parse with `json.loads`).

### Core tools

Generic platform primitives: run any Actor or task and fetch dataset items. Available as the convenience list `APIFY_CORE_TOOLS`:

- `ApifyRunActorTool`: start any Actor, return run metadata
- `ApifyGetDatasetItemsTool`: fetch items from a dataset by ID
- `ApifyRunActorAndGetDatasetTool`: run + fetch in one call
- `ApifyScrapeUrlTool`: single URL to markdown
- `ApifyRunTaskTool`: run a saved Actor task
- `ApifyRunTaskAndGetDatasetTool`: task run + fetch in one call

```python
import os, json
from langchain_apify import ApifyRunActorAndGetDatasetTool

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

result = ApifyRunActorAndGetDatasetTool().invoke({
    "actor_id": "apify/python-example",
    "run_input": {"first_number": 2, "second_number": 3},
})
print(json.loads(result))
```

### Search & crawling tools

Web search, maps, video, e-commerce, and content crawling. Available as `APIFY_SEARCH_TOOLS`:

- `ApifyGoogleSearchTool`: Google search results
- `ApifyWebCrawlerTool`: multi-page website crawler
- `ApifyRAGWebBrowserTool`: search + fetch top results in one call
- `ApifyGoogleMapsTool`: places, reviews, business details
- `ApifyYouTubeScraperTool`: videos, channels, metadata
- `ApifyEcommerceScraperTool`: product pages and category listings

```python
import os, json
from langchain_apify import ApifyGoogleSearchTool

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

result = ApifyGoogleSearchTool().invoke({
    "query": "langchain apify integration",
    "max_results": 5,
})
print(json.loads(result))
```

### Social media tools

Instagram, LinkedIn, Twitter/X, TikTok, and Facebook. Available as `APIFY_SOCIAL_TOOLS`:

- `ApifyInstagramScraperTool`: profiles, hashtags, posts, comments
- `ApifyLinkedInProfilePostsTool`: posts from a LinkedIn profile
- `ApifyLinkedInProfileSearchTool`: keyword search for profiles
- `ApifyLinkedInProfileDetailTool`: full profile detail
- `ApifyTwitterScraperTool`: tweets and users
- `ApifyTikTokScraperTool`: videos, users, hashtags
- `ApifyFacebookPostsScraperTool`: public page posts

```python
import os, json
from langchain_apify import ApifyInstagramScraperTool

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

result = ApifyInstagramScraperTool().invoke({
    "search_type": "user",
    "search_query": "apify",
    "max_results": 3,
})
print(json.loads(result))
```

### Using tools with an agent

Each convenience list lets you bind a whole tool family to an agent in one line. Don't bind all tools at once. Most LLMs lose routing accuracy past ~8 tools, so pick the family the agent actually needs.

```python
import os
from langchain_apify import APIFY_SEARCH_TOOLS
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

model = ChatOpenAI(model="gpt-5.4-mini")
tools = [tool_cls() for tool_cls in APIFY_SEARCH_TOOLS]
agent = create_react_agent(model, tools)

for chunk in agent.stream(
    {"messages": [("human", "search the web for what Apify Actors are")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
```

### `ApifyActorsTool`: wrap any Actor by ID

For Actors without a dedicated wrapper above, `ApifyActorsTool` builds an input schema from the Actor's build at construction time and exposes it as a generic LangChain tool:

```python
import os
from langchain_apify import ApifyActorsTool

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

tool = ApifyActorsTool("apify/rag-web-browser")
result = tool.invoke(input={
    "run_input": {"query": "what is an Apify Actor?", "maxResults": 3},
})
```

## Retriever

`ApifySearchRetriever` is a `BaseRetriever` over `apify/rag-web-browser` for RAG pipelines. Each result becomes a LangChain `Document` with `metadata['source']`, `metadata['title']`, and any additional fields the Actor returns.

```python
import os
from langchain_apify import ApifySearchRetriever

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

retriever = ApifySearchRetriever(max_results=3)
docs = retriever.invoke("what is web scraping")
for doc in docs:
    print(doc.metadata["source"], "-", doc.metadata.get("title"))
```

## Document loaders

> **⚠️ Note for Actor Developers**: If you're building an Apify Actor, use `Actor.open_dataset()` from the Apify SDK instead of these loaders. See the [Note for Apify Actor developers](#note-for-apify-actor-developers) section for details.

### `ApifyCrawlLoader`

Active crawler that wraps `apify/website-content-crawler`. Crawls a seed URL and returns each page as a `Document` with `metadata = {"source", "title", "crawl_depth"}`. Implements `lazy_load()` for streaming and `load()` for the eager collection.

```python
import os
from langchain_apify import ApifyCrawlLoader

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

loader = ApifyCrawlLoader(
    url="https://docs.apify.com",
    max_crawl_pages=5,
    max_crawl_depth=1,
)
documents = loader.load()
```

### `ApifyDatasetLoader`

Loads an existing Apify dataset by ID and maps items to `Document` objects via a user-supplied function. Useful when you have a dataset from a previous run and want to reshape it for downstream LangChain steps.

```python
import os
from langchain_apify import ApifyDatasetLoader
from langchain_core.documents import Document

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

loader = ApifyDatasetLoader(
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"],
        metadata={"source": item["url"]},
    ),
)
```

## Wrappers

`ApifyWrapper` is a higher-level facade that runs an Actor (or task) and returns an `ApifyDatasetLoader` over the result dataset. Useful when you want to run an Actor programmatically and process the results in LangChain in a single chain.

Methods:

- `call_actor` / `acall_actor`: run an Actor and return a loader for the results.
- `call_actor_task` / `acall_actor_task`: run a saved Actor task and return a loader for the results.

```python
import os
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

os.environ["APIFY_TOKEN"] = "YOUR_APIFY_TOKEN"

apify = ApifyWrapper()

loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://python.langchain.com/docs/get_started/introduction"}],
        "maxCrawlPages": 10,
        "crawlerType": "cheerio",
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "",
        metadata={"source": item["url"]},
    ),
)
documents = loader.load()
```

For more information, see the [Apify LangChain integration documentation](https://docs.apify.com/platform/integrations/langchain).

## Note for Apify Actor developers

**If you are building an Apify Actor that will run on the Apify platform**, you should **NOT** use this package for dataset loading. Instead:

**Use the Apify Actor SDK directly** with `Actor.open_dataset()`
**Do NOT use** `ApifyDatasetLoader` from this package

### Why?

1. **Security & permissions**: Actors should run with `LIMITED_PERMISSIONS` and use scoped tokens that grant access only to specific resources. The Actor SDK's `Actor.open_dataset()` method respects these scoped tokens.
2. **Best practices**: Using the Actor SDK is the proper way to access Apify resources within an Actor runtime environment.
3. **No external dependencies**: Your Actor doesn't need to depend on `langchain-apify` for basic dataset operations.

### Example: Loading dataset in an Actor

```python
from apify import Actor
from langchain_core.documents import Document

async def main():
    async with Actor:
        # Get dataset ID from input or integration payload
        dataset_id = Actor.get_input().get("datasetId")

        # Open dataset using Actor SDK (respects LIMITED_PERMISSIONS)
        dataset = await Actor.open_dataset(name=dataset_id)

        # Transform items to Documents
        documents = []
        async for item in dataset.iterate_items():
            doc = Document(page_content=item.get("text", ""), metadata={"url": item.get("url")})
            documents.append(doc)
```

### When to use langchain-apify

This package is designed for:
- External scripts and applications that need to access Apify from outside the Actor runtime
- LangChain agents that use Apify Actors as tools
- Data processing pipelines that consume Apify datasets

It is **NOT** designed for:
- Code running inside an Apify Actor (use Actor SDK instead)

## Contributing

For local setup (Poetry install, running tests and linting), see [DEVELOPMENT.md](DEVELOPMENT.md).
For PR scope, commit message conventions, and review expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).
