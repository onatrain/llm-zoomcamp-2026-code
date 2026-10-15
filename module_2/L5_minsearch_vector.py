from pprint import pprint

from L3_embeddings_dataset import X, faq_items, model
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, faq_items)

query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(query_vector, num_results=5)

pprint(results)

results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)

print("\n", "*" * 100, "\n")
pprint(results)
