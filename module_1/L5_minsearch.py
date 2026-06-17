from L4_read_faq import faq_items
from minsearch import Index  # type: ignore


def search(question: str, course: str = 'llm-zoomcamp') -> list[dict[str, str]]:
    # Con boost_dict se puede indicar cuál campo tiene más peso en el ranking
    return index.search(
        question,
        boost_dict={'question': 2.0, 'section': 1.0, 'answer': 0.5},  # default boost = 1
        filter_dict={'course': course},
        num_results=5
    )


# minsearch emula a ElasticSearch para búsquedas aproximadas en pequeños datasets de texto
# text_fields son los campos que se pueden usar para hacer búsquedas aproximadas
# keyword_fields son campos sobre los que se necesita una búsqueda exacta. Se usan para restringir el rango de búsqueda.
index = Index(
    text_fields=['question', 'section', 'answer'],
    keyword_fields=['course'],
)

index.fit(faq_items)

question = "I just discovered the course. Can I join now?"

search_results = search(question)

