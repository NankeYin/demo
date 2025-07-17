from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


class VectorStore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_vector_store()

    def load_vector_store(self):
        vector_dir = "D:\\demo\\demo\\processed_data\\faiss_index"

        self.embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': self.device}
        )

        self.db = FAISS.load_local(
            vector_dir,
            self.embedder,
            allow_dangerous_deserialization=True
        )

    def search(self, query: str, k: int = 3):
        docs = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]