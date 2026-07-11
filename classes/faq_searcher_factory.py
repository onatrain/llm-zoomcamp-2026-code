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
                path = config.local_index_path
                if path is None or not path.strip():
                    raise ValueError("Local DB path must be provided")
                match config.vector_engine:
                    case "sentence":
                        from classes.local_vector_faq import LocalVectorFAQ
                        from classes.sentence_embedder import SentenceEmbedder
                        index = LocalVectorFAQ(SentenceEmbedder(), db_path=path)
                    case None:
                        from classes.local_faq import LocalFAQ
                        index = LocalFAQ(db_path=path)
                    case _:
                        raise ValueError(f"Unknown vector engine: {config.vector_engine}")
            case _:
                raise ValueError(f"Unknown index type: {config.index_type}")

        index.open()
        faq_searcher = FAQSearcher(index)

        return faq_searcher
