from classes.module_1.faq import FAQ
from classes.module_1.faq_searcher import FAQSearcher
from classes.module_1.in_memory_faq import InMemoryFAQ
from classes.module_1.local_faq import LocalFAQ
from classes.module_1.searcher_config import SearcherConfig


class FAQSearcherFactory:
    def create(self, config: SearcherConfig) -> FAQSearcher:
        index: FAQ | None = None

        match config.index_type:
            case "in_memory":
                index = InMemoryFAQ()
            case "local":
                path = config.local_index_path
                if path is not None and path.strip():
                    index = LocalFAQ(path)
                else:
                    raise ValueError("Local DB path must be provided")
            case _:
                raise ValueError(f"Unknown index_type: {config.index_type}")

        index.open()
        faq_searcher = FAQSearcher(index)

        return faq_searcher
