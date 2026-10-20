from L4_read_faq import faq_items
from minsearch import Index


def search(question: str, course: str = "llm-zoomcamp") -> list[dict[str, str]]:
    """Search FAQ items using approximate matching with boosts.

    Args:
        question: The search query text.
        course: The course slug to filter results by.

    Returns:
        A list of matching FAQ entries as dictionaries.
    """
    # With boost_dict you can indicate which field has more weight in the ranking
    results: list[dict[str, str]] = index.search(
        question,
        boost_dict={
            "question": 2.0,
            "section": 1.0,
            "answer": 0.5,
        },  # default boost = 1
        filter_dict={"course": course},
        num_results=5,
    )
    return results


# minsearch emulates ElasticSearch for approximate search over small text datasets
# text_fields are the fields that can be used for approximate search
# keyword_fields are fields that require an exact search. They are used to restrict the search range.
index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
)

index.fit(faq_items)

question = "I just discovered the course. Can I join now?"

search_results = search(question)
