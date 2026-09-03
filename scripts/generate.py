import ollama
from retrival import retrieve

client = ollama.Client(host="http://127.0.0.1:11434")

PROMPT_TEMPLATE="""
You are a helpful assistant. Use the context below to answer the question.
Synthesize and summarize the relevant information into a clear answer -
you don't need an exact matching sentence, just use what's relevant.
Only say you don't know if the context is genuinely unrelated to the question.
Context:
{context}

Question:
{question}

Answer: 
"""

def answer_question(question, k=3):
    results = retrieve(question, k)

    context = "\n\n---\n\n".join(chunk for chunk, source, distance in results)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    return answer

if __name__ == "__main__":
    question = input("Please ask a question: ")
    answer = answer_question(question)

    print("\n--- Answer ---")
    print(answer)
