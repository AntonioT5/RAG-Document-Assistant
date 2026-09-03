from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.", 
    "The stock market crashed today.",
]

embeddings = model.encode(sentences)
print("Shape of the embedding: ", embeddings[0].shape)

sim_1 = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity between sentence 1 and 2: {sim_1.item():.4f}")

sim_2 = cos_sim(embeddings[0], embeddings[2])
print(f"Similarity between sentence 1 and 3: {sim_2.item():.4f}")
