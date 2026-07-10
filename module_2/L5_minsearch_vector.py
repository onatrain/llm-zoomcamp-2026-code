from minsearch import VectorSearch

from L3_embeddings_dataset import faq_items, model, X


vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, faq_items)

query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(query_vector, num_results=5)

from pprint import pprint

pprint(results)

results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)

print("\n", "*" * 100, "\n")
pprint(results)
