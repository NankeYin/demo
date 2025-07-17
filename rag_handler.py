import requests
from vector_store import VectorStore


class RAGHandler:
    def __init__(self):
        self.vector_store = VectorStore()
        self.ollama_url = "http://localhost:11434/api/generate"

    def get_answer(self, question: str, max_results: int = 3):
        # 1. Retrieve relevant context
        context_docs = self.vector_store.search(question, k=max_results)
        context = "\n\n".join(context_docs)

        # 2. Construct LLM prompt
        prompt = f"""Based on the following banking terminology knowledge base, answer the user's question:

        # Knowledge Base Context:
        {context}

        # User Question:
        {question}

        # Answering Requirements:
        - Respond in professional yet accessible language
        - Politely decline if unrelated to banking terminology
        - Cite source terms at the end of your response
        - Use bullet points for multiple answers
        - No more than 300 words
        - All language should be in English
        """

        # 3. Generate answer via LLM
        payload = {
            "model": "qwen:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300
            }
        }

        response = requests.post(self.ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()

        print(f"LLM response: {result['response']}")

        return {
            "answer": result['response'].strip(),
            "sources": [doc.split("\n")[0].replace("Term: ", "") for doc in context_docs],
            "context": context_docs
        }