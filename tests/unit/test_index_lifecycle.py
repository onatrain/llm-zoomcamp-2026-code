"""Uniform lifecycle contract of every concrete index implementation.

The same scenarios run against the five concrete indexes (``InMemoryIndex``,
``InMemoryVectorIndex``, ``LocalDbIndex``, ``LocalVectorDbIndex`` and
``PgvectorIndex``). External-resource stores are exercised with real SQLite
files and a stubbed psycopg connection, so the suite stays offline while
proving that:

- ``open``/``close``/``is_open`` behave the same everywhere;
- use-while-closed raises the index own ``closed_message`` instead of silently
  doing nothing;
- finalizers exist only where there is an external resource to close, and
  reopening releases the previous resource.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sqlitesearch import TextSearchIndex, VectorSearchIndex

from classes import local_db_index, local_vector_db_index, pgvector_index
from classes.in_memory_index import InMemoryIndex
from classes.in_memory_vector_index import InMemoryVectorIndex
from classes.local_db_index import LocalDbIndex
from classes.local_vector_db_index import LocalVectorDbIndex
from classes.managed_resource import ManagedResource
from classes.pgvector_index import PgvectorIndex

_DSN = "postgresql://postgres:postgres@localhost:5432/faq"
_VECTOR = np.array([1.0, 0.0, 0.0])


class _FakeConnection:
    """Psycopg connection double recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_connect(conninfo: str, row_factory: Any = None) -> _FakeConnection:
    return _FakeConnection()


_FakePsycopg = SimpleNamespace(connect=_fake_connect)


class _RecordingTextSearchIndex(TextSearchIndex):
    """Real SQLite keyword index that counts its close calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class _RecordingVectorSearchIndex(VectorSearchIndex):
    """Real SQLite vector index that counts its close calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _build_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManagedResource[Any]:
    return InMemoryIndex()


def _build_in_memory_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManagedResource[Any]:
    return InMemoryVectorIndex()


def _build_local_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManagedResource[Any]:
    return LocalDbIndex(str(tmp_path / "faq.db"))


def _build_local_vector_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManagedResource[Any]:
    return LocalVectorDbIndex(str(tmp_path / "vector_faq.db"))


def _build_pgvector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManagedResource[Any]:
    monkeypatch.setattr(pgvector_index, "psycopg", _FakePsycopg)
    return PgvectorIndex(_DSN, dimension=3)


def _search_keyword(index: ManagedResource[Any]) -> object:
    return index.search("anything", num_results=1)


def _search_vector(index: ManagedResource[Any]) -> object:
    return index.search(_VECTOR, num_results=1)


@dataclass(frozen=True)
class IndexCase:
    """One concrete index and how to drive it."""

    name: str
    build: Callable[[Path, pytest.MonkeyPatch], ManagedResource[Any]]
    search: Callable[[ManagedResource[Any]], object]
    external_resource: bool


INDEX_CASES = [
    IndexCase("in_memory", _build_in_memory, _search_keyword, False),
    IndexCase("in_memory_vector", _build_in_memory_vector, _search_vector, False),
    IndexCase("local_db", _build_local_db, _search_keyword, True),
    IndexCase("local_vector_db", _build_local_vector_db, _search_vector, True),
    IndexCase("pgvector", _build_pgvector, _search_vector, True),
]


@pytest.fixture(params=INDEX_CASES, ids=[case.name for case in INDEX_CASES])
def case(request: pytest.FixtureRequest) -> IndexCase:
    return request.param


@pytest.fixture
def index(
    case: IndexCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ManagedResource[Any]]:
    store = case.build(tmp_path, monkeypatch)
    yield store
    store.close()


def test_a_fresh_index_starts_closed(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    assert index.is_open is False


def test_every_index_declares_a_closed_message(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    assert index.closed_message


def test_open_and_close_toggle_is_open(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    index.open()
    assert index.is_open is True

    index.close()
    assert index.is_open is False


def test_close_is_idempotent(index: ManagedResource[Any], case: IndexCase) -> None:
    index.open()

    index.close()
    index.close()

    assert index.is_open is False


def test_reopening_keeps_the_index_open(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    index.open()
    first_finalizer = index._finalizer

    index.open()

    assert index.is_open is True
    if case.external_resource:
        assert index._finalizer is not None
        assert first_finalizer is not None
        assert first_finalizer is not index._finalizer
        assert first_finalizer.alive is False
    else:
        assert index._finalizer is None


def test_the_context_manager_closes_the_index(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    with index as handle:
        assert handle is index
        handle.open()
        assert index.is_open is True

    assert index.is_open is False


def test_finalizers_exist_only_for_external_resources(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    index.open()

    if case.external_resource:
        assert index._finalizer is not None
        assert index._finalizer.alive is True

        index.close()
        assert index._finalizer is None
    else:
        assert index._finalizer is None
        assert index._resource is None


def test_size_while_closed_raises_the_closed_message(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(index.closed_message)):
        index.size()


def test_search_while_closed_raises_the_closed_message(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(index.closed_message)):
        case.search(index)


def test_operations_after_close_raise_the_closed_message(
    index: ManagedResource[Any], case: IndexCase
) -> None:
    index.open()
    index.close()

    with pytest.raises(RuntimeError, match=re.escape(index.closed_message)):
        index.size()
    with pytest.raises(RuntimeError, match=re.escape(index.closed_message)):
        case.search(index)


def test_reopening_local_db_closes_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_db_index, "TextSearchIndex", _RecordingTextSearchIndex)
    index = LocalDbIndex(str(tmp_path / "faq.db"))

    index.open()
    first = index._index
    index.open()

    assert isinstance(first, _RecordingTextSearchIndex)
    assert first.close_calls == 1
    assert index._index is not first

    index.close()


def test_reopening_local_vector_db_closes_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        local_vector_db_index, "VectorSearchIndex", _RecordingVectorSearchIndex
    )
    index = LocalVectorDbIndex(str(tmp_path / "vector_faq.db"))

    index.open()
    first = index._index
    index.open()

    assert isinstance(first, _RecordingVectorSearchIndex)
    assert first.close_calls == 1
    assert index._index is not first

    index.close()


def test_reopening_pgvector_closes_the_previous_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_FakeConnection] = []

    def connect(conninfo: str, row_factory: Any = None) -> _FakeConnection:
        connection = _FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(pgvector_index, "psycopg", SimpleNamespace(connect=connect))
    index = PgvectorIndex(_DSN, dimension=3)

    index.open()
    first = connections[0]
    index.open()

    assert len(connections) == 2
    assert first.closed is True
    assert index.is_open is True

    index.close()
    assert connections[1].closed is True
