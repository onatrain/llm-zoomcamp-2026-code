from pprint import pprint

from classes.sentence_embedder import load_transformer

model = load_transformer()

d = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
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

pprint(dv)
