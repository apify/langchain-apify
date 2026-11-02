from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

from langchain_apify.document_loaders import ApifyCrawlLoader, ApifyDatasetLoader
from langchain_apify.retrievers import ApifySearchRetriever
from langchain_apify.tools import (
    APIFY_CORE_TOOLS,
    APIFY_SEARCH_TOOLS,
    APIFY_SOCIAL_TOOLS,
    APIFY_TRANSCRIPT_TOOLS,
    ApifyActorsTool,
    ApifyEcommerceScraperTool,
    ApifyFacebookAdsTranscriptTool,
    ApifyFacebookPostsScraperTool,
    ApifyGetDatasetItemsTool,
    ApifyGoogleMapsTool,
    ApifyGoogleSearchTool,
    ApifyInstagramScraperTool,
    ApifyLinkedInProfileDetailTool,
    ApifyLinkedInProfilePostsTool,
    ApifyLinkedInProfileSearchTool,
    ApifyMediaTranscriberTool,
    ApifyRAGWebBrowserTool,
    ApifyRunActorAndGetDatasetTool,
    ApifyRunActorTool,
    ApifyRunTaskAndGetDatasetTool,
    ApifyRunTaskTool,
    ApifyScrapeUrlTool,
    ApifyTikTokScraperTool,
    ApifyTwitterScraperTool,
    ApifyWebCrawlerTool,
    ApifyYouTubeScraperTool,
    ApifyYouTubeTranscriptTool,
)
from langchain_apify.wrappers import ApifyWrapper

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

try:
    __version__ = metadata.version(__package__) if __package__ is not None else ''
except metadata.PackageNotFoundError:
    __version__ = ''
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    # Existing components (backward-compatible)
    'ApifyActorsTool',
    'ApifyCrawlLoader',
    'ApifyDatasetLoader',
    'ApifySearchRetriever',
    'ApifyWrapper',
    # Core generic tools
    'ApifyGetDatasetItemsTool',
    'ApifyRunActorAndGetDatasetTool',
    'ApifyRunActorTool',
    'ApifyRunTaskAndGetDatasetTool',
    'ApifyRunTaskTool',
    'ApifyScrapeUrlTool',
    # Social media Actor tools
    'ApifyFacebookPostsScraperTool',
    'ApifyInstagramScraperTool',
    'ApifyLinkedInProfileDetailTool',
    'ApifyLinkedInProfilePostsTool',
    'ApifyLinkedInProfileSearchTool',
    'ApifyTikTokScraperTool',
    'ApifyTwitterScraperTool',
    # Search & crawling tools
    'ApifyGoogleSearchTool',
    'ApifyWebCrawlerTool',
    'ApifyRAGWebBrowserTool',
    'ApifyGoogleMapsTool',
    'ApifyYouTubeScraperTool',
    'ApifyEcommerceScraperTool',
    # Transcription Actor tools
    'ApifyFacebookAdsTranscriptTool',
    'ApifyYouTubeTranscriptTool',
    'ApifyMediaTranscriberTool',
    # Tool group lists
    'APIFY_CORE_TOOLS',
    'APIFY_SOCIAL_TOOLS',
    'APIFY_SEARCH_TOOLS',
    'APIFY_TRANSCRIPT_TOOLS',
    # Meta
    '__version__',
]
