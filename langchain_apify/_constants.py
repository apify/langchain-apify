"""Shared default values for Apify tool and client parameters.

Centralized so every tunable default lives in one place and is imported by the
client, tools, retriever, and loaders rather than reaching into one another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_apify._types import CrawlerType

# Default timeouts (seconds), by operation class.
_DEFAULT_RUN_TIMEOUT_SECS = 300
_DEFAULT_SCRAPE_TIMEOUT_SECS = 120
_DEFAULT_SOCIAL_TIMEOUT_SECS = 600
_DEFAULT_TRANSCRIPT_TIMEOUT_SECS = 600

# Default result / page limits, by operation.
_DEFAULT_DATASET_ITEMS_LIMIT = 100
_DEFAULT_MAX_CRAWL_PAGES = 10
_DEFAULT_MAX_CRAWL_DEPTH = 1
_DEFAULT_GOOGLE_MAX_RESULTS = 10
_DEFAULT_RAG_MAX_RESULTS = 5
_DEFAULT_SOCIAL_RESULTS_LIMIT = 20
_DEFAULT_LINKEDIN_SEARCH_MAX_RESULTS = 10
_DEFAULT_GOOGLE_MAPS_MAX_RESULTS = 10
_DEFAULT_YOUTUBE_MAX_RESULTS = 10
_DEFAULT_ECOMMERCE_MAX_RESULTS = 20
# Transcription Actors charge per delivered item, so the family defaults to a
# smaller batch than the scraping families.
_DEFAULT_TRANSCRIPT_RESULTS_LIMIT = 10

# Upper-bound clamp ceilings applied by _ApifyGenericTool to LLM-supplied
# values. The Pydantic Field defaults on the base class reference these so
# the schema descriptions can interpolate the same numbers, keeping the
# clamp documentation truthful even if the cap moves.
_MAX_TIMEOUT_SECS_CAP = 600
_MAX_MEMORY_MBYTES_CAP = 32768
_MAX_ITEMS_CAP = 1000
_MAX_CRAWL_DEPTH_CAP = 5

# The apify/rag-web-browser Actor rejects maxResults > 100 at runtime. This
# limit is enforced by the Actor, not declared in its input schema (which only
# carries a default), so it cannot be derived by schema introspection and must
# be tracked here by hand. Applied by ApifyRAGWebBrowserTool and
# ApifySearchRetriever, both of which wrap that Actor.
_RAG_MAX_RESULTS_CAP = 100

# Default crawler engine for the website-content-crawler Actor.
_DEFAULT_CRAWLER_TYPE: CrawlerType = 'cheerio'
