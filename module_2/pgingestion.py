import os

import dotenv

from classes.pgvector_index import PgvectorIndex
from classes.semantical_faq import SemanticalFAQ
from classes.sentence_embedder import SentenceEmbedder

dotenv.load_dotenv()

dsn = os.getenv("DATABASE_URL")
if not dsn or not dsn.strip():
    raise ValueError("DATABASE_URL must be set")

embedder = SentenceEmbedder()
faq = SemanticalFAQ(
    PgvectorIndex(dsn, dimension=embedder.embedding_dimension), embedder
)

faq.build_index()

print("The new index contains", faq.count, "documents.")

faq.close()