import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.index.add(np.array(embeddings))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]), top_k
        )

        results = []

        for i in indices[0]:
            if i < len(self.text_chunks):
                results.append({
                    "chunk_id": i,
                    "text": self.text_chunks[i]
                })

        return results