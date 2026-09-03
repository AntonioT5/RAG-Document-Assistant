from load_pdf import load_pdf


def chuck_text(text, chunk_size=1000, overlap=150):
    """
    Splits text into overllapping chunks:
    chunk_size: how many characters per chunk
    overlap: how many characters from the end
    of one chunk are repeated at the start of the next chunk
    """
    chunks = []
    start = 0

    while start<len(text):
        end = start+chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size-overlap

    return chunks

if __name__ == "__main__":
    docs = load_pdf("documents")

    for doc in docs:
        chunks = chuck_text(doc["text"])
        print(f"{doc['filename']}: {len(doc['text'])} characters -> {len(chunks)} chunks")
        print("--- First chunk preview ---")
        print(chunks[0][:200])
        print()