import os

os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", token=False)

d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)

print("Document vector dimensions:", dv.shape)

q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)

print("Distance between the enrollment question and the document")
print(v1.dot(dv))
print(dv.dot(v1))

q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)

print("Distance between the docker question and the document")
print(v2.dot(dv))
print(dv.dot(v2))
print()

from pprint import pprint
pprint(dv)

