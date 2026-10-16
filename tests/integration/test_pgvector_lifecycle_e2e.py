"""End-to-end lifecycle tests of PgvectorIndex against a real PostgreSQL.

These tests only open and close connections: they never create the documents
table and never read the FAQ corpus, so the remote site is not hit and the
database in ``DATABASE_URL`` is not modified (the module-scoped ``pg_test_dsn``
fixture creates a throwaway database).
"""

import pytest

from classes.pgvector_index import PgvectorIndex


def test_operations_on_a_closed_store_raise(pg_test_dsn: str) -> None:
    store = PgvectorIndex(pg_test_dsn, dimension=3)

    assert store.is_open is False

    with pytest.raises(RuntimeError, match="closed"):
        store.size()

    store.close()
    assert store.is_open is False


def test_reopening_closes_the_previous_connection(pg_test_dsn: str) -> None:
    store = PgvectorIndex(pg_test_dsn, dimension=3)

    store.open()
    first = store._resource
    assert store.is_open is True
    assert store._finalizer is not None
    assert store._finalizer.alive is True

    store.open()

    assert first is not None
    assert first.closed is True
    assert store.is_open is True

    store.close()

    assert store.is_open is False
    assert store._finalizer is None
