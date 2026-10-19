"""Unit tests for the optional search parameter normalizer."""


from utils.normalize import normalize_index_param


def test_none_returns_none() -> None:
    assert normalize_index_param(None) is None


def test_empty_mapping_returns_none() -> None:
    assert normalize_index_param({}) is None


def test_none_values_are_dropped() -> None:
    assert normalize_index_param({"a": None, "b": 2}) == {"b": 2}


def test_all_none_returns_none() -> None:
    assert normalize_index_param({"a": None, "b": None}) is None


def test_valid_mapping_is_copied() -> None:
    original = {"a": 1, "b": "x"}
    result = normalize_index_param(original)

    assert result == original
    assert result is not original
