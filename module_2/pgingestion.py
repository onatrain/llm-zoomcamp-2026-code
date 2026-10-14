from classes.pgvector_repository import PgvectorRepository
from classes.sentence_embedder import SentenceEmbedder

db = PgvectorRepository()
db.connect()
db.db_setup()

embedder = SentenceEmbedder()

print(db)

db.close()