import chromadb
from sentence_transformers import SentenceTransformer
from load_pdf import load_pdf
from chuncking import chuck_text

model = SentenceTransformer("all-MiniLM-L6-v2")

'''
Save the ChromaDB to hard disk and create/get a collection
named "documents".

A collection is similar to a table in traditional database.
'''
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="documents")

def build_index(document_dir="documents"):
    '''
    Load all PDF, then makes chunks of that documents,
    creats vectors and store them in chroma database
    '''
    docs = load_pdf(document_dir)

    for doc in docs:
        chunks = chuck_text(doc["text"])
        embeddings = model.encode(chunks).tolist()

        ids = [f"{doc['filename']}-{i}" for i in range(len(chunks))]

        metadates = [{"source": doc["filename"]} for _ in chunks]

        collection.add(
            ids = ids,
            embeddings= embeddings,
            documents=chunks,
            metadatas=metadates
        )

    print(f"Indexed {collection.count()} chunks total.")

if __name__ == "__main__":
    build_index()