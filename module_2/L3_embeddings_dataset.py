import numpy as np
from tqdm.auto import tqdm

from classes.raw_faq import RawFaq
from classes.sentence_embedder import load_transformer

# Load the model once to create the embeddings of the dataset
model = load_transformer()

# Load the FAQ documents as question + answer
raw_faq = RawFaq()
faq_items = raw_faq.items

texts = [" ".join((d["question"], d["answer"])) for d in faq_items]

# Create the Python matrix (list of lists) with the embedded text vectors in the model's vector space
batch_size = 50

batch_generator = (
    model.encode(texts[i : i + batch_size])
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches")
)

vectors = [vector for batch in batch_generator for vector in batch]

print("Loaded", len(vectors), "documents.")
print("Type of each vector:", type(vectors[0]))  # <class 'numpy.ndarray'>

# Create the numpy matrix (array of arrays) from the embedded text vector matrix
X = np.array(vectors)
