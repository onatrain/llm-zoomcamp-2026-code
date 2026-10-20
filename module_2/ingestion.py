from pathlib import Path

from classes.local_vector_db_index import LocalVectorDbIndex
from classes.semantical_faq import SemanticalFAQ
from classes.sentence_embedder import SentenceEmbedder

current_dir = Path(__file__).parent

local_vector_faq = SemanticalFAQ(
    LocalVectorDbIndex(str(current_dir / "data/vector_faq.db")), SentenceEmbedder()
)

local_vector_faq.build_index()

print("The created index contains", local_vector_faq.count, "documents.")
