from typing import Any, Sequence

import psycopg
import psycopg.sql
from numpy.typing import NDArray
from psycopg import Connection
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row

from classes.managed_resource import ManagedResource
from classes.vector_index import SearchResult, Vector, VectorIndex

_INSERT_COLUMNS = ("course", "section", "question", "answer")
_INSERT_SQL = """
    INSERT INTO documents (course, section, question, answer, embedding)
    VALUES (%s, %s, %s, %s, %s::vector)
"""
_SEARCH_SQL = """
    SELECT course, section, question, answer,
           embedding <=> %s::vector AS distance
    FROM documents
    {where}
    ORDER BY distance
    LIMIT %s
"""
_COUNT_SQL = "SELECT COUNT(*) FROM documents"


def vector_literal(vector: NDArray[Any]) -> str:
    """Format a vector as the text literal accepted by pgvector.

    Args:
        vector: The vector to format.

    Returns:
        The pgvector text representation of the vector.
    """
    return "[" + ",".join(str(value) for value in vector) + "]"


class PgvectorIndex(ManagedResource[Connection[dict[str, Any]]], VectorIndex):
    """Vector store backed by PostgreSQL with the pgvector extension."""

    closed_message = "Database closed. You must connect() it first."

    def __init__(self, dsn: str, *, dimension: int) -> None:
        """Initialize the store.

        Args:
            dsn: The PostgreSQL connection string. Must be provided by the
                caller; this class never reads the environment.
            dimension: The embedding dimension the ``documents`` table will use.
                Must match the dimension of the embedder feeding this store.
                It is required and has no default.

        Raises:
            ValueError: If ``dsn`` is empty.
            ValueError: If the dimension is not a positive integer.
        """
        super().__init__()

        if not dsn or not dsn.strip():
            raise ValueError("A PostgreSQL DSN must be provided")

        self._dsn: str = dsn

        if dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Embedding dimension expected by this store."""
        return self._dimension

    def open(self) -> None:
        """Open the connection to the database.

        Any previously open connection is closed first.

        Raises:
            psycopg.OperationalError: If the database is not reachable.
        """
        if self.is_open:
            self.close()

        connection = psycopg.connect(conninfo=self._dsn, row_factory=dict_row)
        self._register_resource(connection, connection.close)

    def reset(self) -> None:
        """Drop and recreate the ``documents`` table and its HNSW index.

        This operation is destructive: every stored document is deleted.
        """
        conn = self._require_resource()

        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute(
            psycopg.sql.SQL("""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    course TEXT,
                    section TEXT,
                    question TEXT,
                    answer TEXT,
                    embedding vector({dimension})
                )
            """).format(dimension=psycopg.sql.Literal(self._dimension))
        )

        conn.execute("""
            CREATE INDEX IF NOT EXISTS hnsw_documents_ndx ON documents
            USING hnsw (embedding vector_cosine_ops)
        """)

        conn.commit()

    def add(self, document: dict[str, str], vector: Vector) -> None:
        """Insert a single document with its embedding.

        The insert is committed immediately.

        Args:
            document: The document fields to store.
            vector: The embedding of the document.

        Raises:
            KeyError: If a document is missing one of the stored fields.
            RuntimeError: If the documents table does not exist yet.
        """
        conn = self._require_resource()

        self._execute_insert(document, vector)
        conn.commit()

    def _execute_insert(self, document: dict[str, str], vector: Vector) -> None:
        """Execute the insert of one document without committing."""
        values = [document[field] for field in _INSERT_COLUMNS]
        self._execute(_INSERT_SQL, (*values, vector_literal(vector)))

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Run a statement against the open connection.

        Args:
            sql: The statement to run.
            params: The parameters of the statement.

        Returns:
            The cursor of the executed statement.

        Raises:
            RuntimeError: If the store is closed or the documents table does not
                exist yet, which means the index was never built.
        """
        conn = self._require_resource()

        try:
            return conn.execute(sql, params)
        except UndefinedTable as exc:
            raise RuntimeError(
                "The documents table does not exist yet. Build the index first."
            ) from exc

    def _fetch_all(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        """Run a query and return every row as a mapping.

        Args:
            sql: The query to run.
            params: The parameters of the query.

        Returns:
            The rows returned by the query.
        """
        rows: list[dict[str, Any]] = self._execute(sql, params).fetchall()
        return rows

    def add_all(
        self, documents: Sequence[dict[str, str]], vectors: Sequence[Vector]
    ) -> None:
        """Insert many documents with their embeddings in a single transaction.

        Args:
            documents: The documents to store.
            vectors: The embeddings of the documents, aligned with ``documents``.

        Raises:
            ValueError: If the number of documents and vectors differ.
            KeyError: If a document is missing one of the stored fields.
            RuntimeError: If the documents table does not exist yet.
        """
        if len(documents) != len(vectors):
            raise ValueError(
                f"Got {len(documents)} documents and {len(vectors)} vectors, "
                "they must be the same amount"
            )

        conn = self._require_resource()

        for document, vector in zip(documents, vectors, strict=True):
            self._execute_insert(document, vector)

        conn.commit()

    def search(
        self, query_vector: Vector, course: str | None = None, num_results: int = 5
    ) -> list[SearchResult]:
        """Search the store by cosine similarity.

        Args:
            query_vector: The embedding of the query.
            course: Optional course slug to restrict the search to.
            num_results: Maximum number of results to return.

        Returns:
            The matching results, ordered from closest to farthest.

        Raises:
            RuntimeError: If the documents table does not exist yet.
        """
        literal = vector_literal(query_vector)

        if course is None:
            sql = _SEARCH_SQL.format(where="")
            params: tuple[Any, ...] = (literal, num_results)
        else:
            sql = _SEARCH_SQL.format(where="WHERE course = %s")
            params = (literal, course, num_results)

        rows = self._fetch_all(sql, params)

        return [
            SearchResult(
                document={field: row[field] for field in _INSERT_COLUMNS},
                distance=float(row["distance"]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            The document count.

        Raises:
            RuntimeError: If the documents table does not exist yet.
        """
        rows = self._fetch_all(_COUNT_SQL)
        if not rows:
            return 0

        count: int = rows[0]["count"]
        return count

    def close(self) -> None:
        """Close the connection to the database.

        Closing an already closed store is a no-op.
        """
        if self._resource is None:
            return

        conn = self._require_resource()
        conn.close()
        self._release_resource()


PgVectorIndex = PgvectorIndex
