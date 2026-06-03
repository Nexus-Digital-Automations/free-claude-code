"""Public surface of the repo_index subpackage.

Exports the three types optimizer.py and external callers need.
All other modules are internal — import them directly only from within repo_index/.
"""

from .cache_stats import CacheStatsTracker
from .index import LoadedIndex, RepoIndex

__all__ = ["RepoIndex", "LoadedIndex", "CacheStatsTracker"]
