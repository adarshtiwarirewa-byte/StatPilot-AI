import faiss
import numpy as np 


class VectorStore :
    def __init__(self,embedding_dim = 34):
        """
        embedding_dim: all-MiniLM-L6-v2 produces 384-dimensional vectors
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []  # stores {"page":..., "chunk":...} for each vector, in same order


    def add_chunks(self,chunks_with_embeddings):
        """
        Args : chunks with embeddings : list of dicts [{"page" : 1, "chunk" : "...","embedding" : [...]}....]
        
        """

        embeddings = np.array([c["embeddings"] for c in chunks_with_embeddings]).astype('float32')
        self.index.add(embeddings)

        # Store metadata separately , same order as embeddings added

        for c in chunks_with_embeddings:
            self.metadata.append({"page":c["page"],"chunk":c["chunk"]})



    def search(self, query_embedding, top_k=3):
        """
        Args:
            query_embedding: embedding vector of the user's question
            top_k: number of closest chunks to retrieve
        
        Returns:
            list of dicts: [{"page": ..., "chunk": ..., "distance": ...}, ...]
        """

        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector,top_k)

        results = []
        for dist,idx in zip(distances[0],indices[0]):
            if idx == -1 :        # FAISS retuns -1 if not enough results
                continue
            result = self.metadata[idx].copy()
            result["distance"] = float(dist)
            results.append(result)

        return results



if __name__ == "__main__":
    # test (stand alone)
    from pdf_parser import extract_text_from_pdf
    from embedding_engine import chunk_text, embed_chunks,model

    test_pdf_path = r"C:\Users\Adarsh Tiwari\OneDrive\Desktop\IIT IMP Documents\1st sem\Linear Algebra\Assignments\assignment 17.pdf"  # put your text pdf path

    pages = extract_text_from_pdf(test_pdf_path)
    chunks = chunk_text(pages)
    chunks_with_embedd = embed_chunks(chunks)


    store = VectorStore(embedding_dim=384)
    store.add_chunks(chunks_with_embedd)

    test_query = "What does this document talk about?"
    query_embeddings = model.encode(test_query)

    results = store.search(query_embeddings,top_k=3)
    print(len(results))
    print("\n\n")
    print(f"Top {len(results)} matches:\n")
    for r in results:
        print(f"Page {r['page']} (distance: {r['distance']:.4f})")
        print(f"{r['chunk'][:150]}...\n")