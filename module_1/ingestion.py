from pathlib import Path

from classes.lexical_faq import LexicalFAQ
from classes.local_db_index import LocalDbIndex

current_dir = Path(__file__).parent

local_faq = LexicalFAQ(LocalDbIndex(str(current_dir / "data/faq.db")))

local_faq.build_index()

print("The new index contains", local_faq.count, "documents.")
