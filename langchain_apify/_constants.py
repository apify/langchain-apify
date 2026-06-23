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

# Default crawler engine for the website-content-crawler Actor.
_DEFAULT_CRAWLER_TYPE: CrawlerType = 'cheerio'
