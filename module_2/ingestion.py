from pathlib import Path

from classes.local_vector_faq import LocalVectorFAQ
from classes.sentence_embedder import SentenceEmbedder

current_dir = Path(__file__).parent

local_vector_faq = LocalVectorFAQ(SentenceEmbedder(), str(current_dir / "data/vector_faq.db"))

local_vector_faq.build_index()

print("El indice creado contiene", local_vector_faq.count, "documentos.")
