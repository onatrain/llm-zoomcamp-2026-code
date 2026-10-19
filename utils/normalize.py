"""Helpers for normalizing optional search parameters."""

from collections.abc import Mapping
from typing import Any


def normalize_index_param(param: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Drop ``None`` entries and empty mappings from an optional search param.

    Args:
        param: Optional mapping of search parameters (e.g. filters or boosts).

    Returns:
        A new dict without ``None`` values, or ``None`` if the input is ``None``
        or becomes empty.
    """
    if not param:
        return None
    normalized = {key: value for key, value in param.items() if value is not None}
    return normalized or None
