from importlib import metadata

from langchain_apify.document_loaders import ApifyDatasetLoader
from langchain_apify.tools import ApifyActorsTool
from langchain_apify.wrappers import ApifyWrapper

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ''
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    'ApifyActorsTool',
    'ApifyDatasetLoader',
    'ApifyWrapper',
    '__version__',
]
