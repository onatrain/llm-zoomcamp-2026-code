from module_1.classes.faq import FAQ
from module_1.classes.faq_searcher import FAQSearcher
from module_1.classes.in_memory_faq import InMemoryFAQ
from module_1.classes.local_faq import LocalFAQ
from module_1.classes.searcher_config import SearcherConfig


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
