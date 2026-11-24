from __future__ import annotations

from typing import Literal

# Shared Literal aliases for tool/client parameters, centralized so every
# accepted-value set is declared once and reused across schemas and signatures.

# Search & crawling
CrawlerType = Literal['cheerio', 'playwright:adaptive', 'playwright:firefox']
YouTubeSearchType = Literal['search', 'video', 'channel']
EcommerceUrlType = Literal['product', 'category']

# Social media
InstagramSearchType = Literal['user', 'hashtag', 'post', 'comments']
TwitterSearchMode = Literal['search', 'user', 'replies']
TwitterSort = Literal['Latest', 'Top']
TikTokSearchType = Literal['search', 'user', 'hashtag', 'post']
