import os
from pypdf import PdfReader

DOCUMENT_DIR = "documents"

def load_pdf(document_dir):
    '''
    Reads every PDF in the folder and return lists of dicts,
    each holding the full text of one document plus it's file name
    '''
    documents = []

    for file in os.listdir(document_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(document_dir, file)

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            documents.append({
                "filename": file,
                "text": text
            })

    return documents

if __name__ == "__main__":
    docs = load_pdf(DOCUMENT_DIR)
    print(f"Loaded {len(docs)} document(s).")
    for doc in docs:
        print(f"{doc["filename"]}: {len(doc["text"])} characters")