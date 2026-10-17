from __future__ import annotations

# The metadata lives in a sibling collection (``<collection>__index_info``) so it
# never pollutes search results.  Qdrant point ids must be uint or UUID, hence the
# fixed UUID for the single metadata point.
INDEX_INFO_POINT_ID = "00000000-0000-0000-0000-000000000013"


def index_info_collection(collection: str) -> str:
    """Return the sibling collection that holds ``collection``'s metadata."""
    return f"{collection}__index_info"
__all__ = ["INDEX_INFO_POINT_ID", "index_info_collection"]

