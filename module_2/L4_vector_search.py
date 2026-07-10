from L3_embeddings_dataset import texts, model, np, X

# Query embedding
query = "Can I still join the course after the start date?"
v_query = model.encode(query)

# Vector search
scores = X.dot(v_query)  # scores es un array cuyos elementos son los scores (floats)

# Obtiene el score más alto y su documento
idx = np.argmax(scores)
print("The best score found is in pos", idx, "and its value is", scores[idx])

print()

# Obtiene los 5 scores más altos y sus documentos
top5 = np.argsort(-scores)[:5]
print("Los 5 mejores scores son:")
for idx in top5:
    print(scores[idx])
    print(texts[idx])
    print()
