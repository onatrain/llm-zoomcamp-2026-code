from pathlib import Path

from classes.local_faq import LocalFAQ

current_dir = Path(__file__).parent

local_faq = LocalFAQ(str(current_dir / "data/faq.db"))

local_faq.build_index()

print("The new index contains", local_faq.count, "documents.")
