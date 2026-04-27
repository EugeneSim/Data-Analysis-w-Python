"""Singapore HDB resale EDA: ingestion through insights."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("singapore-eda")
except PackageNotFoundError:
    __version__ = "0.0.0"
