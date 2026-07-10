from classes.faq import FAQ
from classes.faq_searcher import FAQSearcher
from classes.searcher_config import SearcherConfig


class FAQSearcherFactory:
    def create(self, config: SearcherConfig) -> FAQSearcher:
        index: FAQ | None = None

        match config.index_type:
            case "in_memory":
                match config.vector_engine:
                    case "sentence":
                        from classes.in_memory_vector_faq import InMemoryVectorFAQ
                        from classes.sentence_embedder import SentenceEmbedder
                        index = InMemoryVectorFAQ(SentenceEmbedder())
                    case None:
                        from classes.in_memory_faq import InMemoryFAQ
                        index = InMemoryFAQ()
                    case _:
                        raise ValueError(f"Unknown vector engine: {config.vector_engine}")
            case "local":
                match config.vector_engine:
                    case None:
                        from classes.local_faq import LocalFAQ
                        path = config.local_index_path
                        if path is not None and path.strip():
                            index = LocalFAQ(path)
                        else:
                            raise ValueError("Local DB path must be provided")
                    case _:
                        raise ValueError(f"Unknown vector engine: {config.vector_engine}")
            case _:
                raise ValueError(f"Unknown index type: {config.index_type}")

        index.open()
        faq_searcher = FAQSearcher(index)

        return faq_searcher
