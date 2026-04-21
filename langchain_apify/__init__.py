from __future__ import annotations

from importlib import metadata

from langchain_apify.document_loaders import ApifyDatasetLoader
from langchain_apify.tools import (
    ApifyActorsTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetItemsTool,
    ApifyRunActorTool,
    ApifyScrapeUrlTool,
)
from langchain_apify.wrappers import ApifyWrapper

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ''
del metadata  # optional, avoids polluting the results of dir(__package__)

# Convenience tool-class lists for selective agent binding.
# Binding all tools at once overwhelms the LLM context window;
# pick the group(s) relevant to your use case.

APIFY_CORE_TOOLS: list[type] = [
    ApifyRunActorTool,
    ApifyGetDatasetItemsTool,
    ApifyRunActorAndGetItemsTool,
    ApifyScrapeUrlTool,
]

__all__ = [
    # Existing components (backward-compatible)
    'ApifyActorsTool',
    'ApifyDatasetLoader',
    'ApifyWrapper',
    # Core generic tools
    'ApifyGetDatasetItemsTool',
    'ApifyRunActorAndGetItemsTool',
    'ApifyRunActorTool',
    'ApifyScrapeUrlTool',
    # Tool group lists
    'APIFY_CORE_TOOLS',
    # Meta
    '__version__',
]
