from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

from langchain_apify._actor_tools import (
    ApifyEcommerceScraperTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyRAGWebBrowserTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
)
from langchain_apify.document_loaders import ApifyCrawlLoader, ApifyDatasetLoader
from langchain_apify.retrievers import ApifySearchRetriever
from langchain_apify.tools import (
    ApifyActorsTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetDatasetTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
)
from langchain_apify.wrappers import ApifyWrapper

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ''
del metadata  # optional, avoids polluting the results of dir(__package__)

# Convenience tool-class lists for selective agent binding.
# Binding all tools at once overwhelms the LLM context window;
# pick the group(s) relevant to your use case.

APIFY_CORE_TOOLS: list[type[BaseTool]] = [
    ApifyRunActorTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyScrapeUrlTool,
    ApifyRunTaskTool,
    ApifyRunTaskAndGetDatasetTool,
]

APIFY_ACTOR_TOOLS: list[type[BaseTool]] = [
    ApifyGoogleSearchTool,
    ApifyWebCrawlerTool,
]

APIFY_SEARCH_TOOLS: list[type[BaseTool]] = [
    ApifyRAGWebBrowserTool,
    ApifyGoogleMapsTool,
    ApifyYouTubeScraperTool,
    ApifyEcommerceScraperTool,
]

__all__ = [
    # Existing components (backward-compatible)
    'ApifyActorsTool',
    'ApifyDatasetLoader',
    'ApifyWrapper',
    # Core generic tools
    'ApifyGetDatasetItemsTool',
    'ApifyRunActorAndGetDatasetTool',
    'ApifyRunActorTool',
    'ApifyRunTaskAndGetDatasetTool',
    'ApifyRunTaskTool',
    'ApifyScrapeUrlTool',
    # Actor-specific tools
    'ApifyGoogleSearchTool',
    'ApifyWebCrawlerTool',
    # Search & Crawling Actor tools
    'ApifyEcommerceScraperTool',
    'ApifyGoogleMapsTool',
    'ApifyRAGWebBrowserTool',
    'ApifyYouTubeScraperTool',
    # Retriever
    'ApifySearchRetriever',
    # Loaders
    'ApifyCrawlLoader',
    # Tool group lists
    'APIFY_ACTOR_TOOLS',
    'APIFY_CORE_TOOLS',
    'APIFY_SEARCH_TOOLS',
    # Meta
    '__version__',
]
