from L3_embeddings_dataset import texts, model, np, X

# Query embedding
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
print("Type of a embedded query:", type(v_query))  # <class 'numpy.ndarray'>

# Vector search
scores = X.dot(v_query)  # scores is an array whose elements are the scores (floats)

# Get the highest score and its document
idx = np.argmax(scores)
print("The best score found is in pos", idx, "and its value is", scores[idx])

print()

# Get the top 5 scores and their documents
top5 = np.argsort(-scores)[:5]
print("The 5 best scores are:")
for idx in top5:
    print(scores[idx])
    print(texts[idx])
    print()
