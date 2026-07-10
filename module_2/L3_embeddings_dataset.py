import os

import numpy as np
from tqdm.auto import tqdm

from classes.raw_faq import RawFaq

# Instancia SentenceTransformer y carga el modelo para crear embeddings
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

# Carga los documentos de la FAQ como pregunta + respuesta
raw_faq = RawFaq()
faq_items = raw_faq.items

texts = [" ".join((d["question"], d["answer"])) for d in faq_items]

# Crea la matriz python (lista de listas) con los vectores de textos embebidos en el vector space del modelo
batch_size = 50

batch_generator = (
    model.encode(texts[i : i + batch_size])
    for i in tqdm(range(0, len(texts), batch_size), desc="Procesando batches")
)

vectors = [vector for batch in batch_generator for vector in batch]

print("Cargados", len(vectors), "documentos.")

# Crea la matriz numpy (array de arrays) a partir de la matriz de vectores de textos embebidos
X = np.array(vectors)
