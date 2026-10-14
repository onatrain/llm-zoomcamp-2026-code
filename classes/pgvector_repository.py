import os, dotenv
import weakref
from typing import Any

import psycopg
from psycopg import Connection
from tqdm import tqdm
from numpy.typing import NDArray

dotenv.load_dotenv()


class PgvectorRepository:
    def __init__(self) -> None:
        self._db: Connection | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None

    def connect(self) -> None:
        if self._db is not None:
            self._db.close()

        self._db = psycopg.connect(conninfo=os.getenv("DATABASE_URL"))

        self._finalizer = weakref.finalize(self, self._db.close)

    def bulk_insert(self, documents, vectors: NDArray, fieldnames: list[str]) -> None:
        if self._db is None:
            raise RuntimeError("Database closed. You must open it first.")

        vec_to_str = lambda vector: "[" + ",".join(str(x) for x in vector) + "]"

        for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
            # doc["course"], doc["section"], doc["question"], doc["answer"]
            values = [doc[f] for f in fieldnames]
            self._db.execute(
                """
                INSERT INTO documents (course, section, question, answer, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (*values, vec_to_str(vec))
            )

        self._db.commit()

    def db_setup(self) -> None:
        if self._db is None:
            raise RuntimeError("Database closed. You must open it first.")

        conn = self._db
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute("""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                course TEXT,
                section TEXT,
                question TEXT,
                answer TEXT,
                embedding vector(384)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS hnsw_documents_ndx ON documents
            USING hnsw (embedding vector_cosine_ops)
        """)

        conn.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
        self._db = None

        # Must detach finalizer to avoid double calling when GC cleans later.
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()
            self._finalizer = None

    def __repr__(self) -> str:
        if self._db is not None:
            info = self._db.info
            rep = f"{info.user}@{info.host}:{info.port}/{info.dbname}"
        else:
            rep = "Postgres database closed"
        return rep
