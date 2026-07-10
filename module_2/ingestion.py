from pathlib import Path

from classes.local_faq import LocalFAQ

current_dir = Path(__file__).parent

local_faq = LocalFAQ(str(current_dir / "data/faq.db"))

local_faq.build_index()

print("El indice creado contiene", local_faq.count, "documentos.")
