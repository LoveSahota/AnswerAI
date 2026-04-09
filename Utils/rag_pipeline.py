import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(query, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:
        context += chunk["text"] + "\n\n"

    prompt = f"""
You are a professional AI document assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"The answer is not available in the uploaded document."

Context:
{context}

Question:
{query}

Provide a clear and structured answer.
"""

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["response"]
    else:
        return "Error generating response."