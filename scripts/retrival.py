import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="documents")

def retrieve(question, k=3):
    """
    Embeds the quetion the same way we embeded the chunks,
    then ask Chroma for the k nearest chunks
    """

    question_embeding = model.encode([question]).tolist()

    result = collection.query(
        query_embeddings=question_embeding,
        n_results=k
    )

    chunks = result["documents"][0]
    sources = result["metadatas"][0]
    distances = result["distances"][0]

    return list(zip(chunks, sources, distances))

if __name__ == "__main__":
    question = input("Please ask a quetion: ")
    result = retrieve(question)

    for i, (chunk, source, distance) in enumerate(result):
        print(f"Result {i+1} (from {source['source']}, distance={distance:.4f})")
        print(chunk[:300])