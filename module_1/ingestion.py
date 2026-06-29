from pathlib import Path

from module_1.classes.local_faq import LocalFAQ

current_dir = Path(__file__).parent

local_faq = LocalFAQ(str(current_dir / "data/faq.db"))

index = local_faq.build_index()

print("El indice creado contiene", index.count(), "documentos.")
