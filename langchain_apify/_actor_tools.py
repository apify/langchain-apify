from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_apify._client import ApifyToolsClient
from langchain_apify.tools import _ApifyGenericTool, _run_meta

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun


# ---------------------------------------------------------------------------
# Search & Crawling tools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Social-media tools
# ---------------------------------------------------------------------------