from utils.embedding_engine import model
from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)



def answer_question(question,vector_store,top_k=3):
    """
    Args:
        question: user's natural language question
        vector_store: an instance of VectorStore class (already populated with chunks)
        top_k: how many chunks to retrieve as context
    
    Returns:
        dict: {"answer": "...", "sources": [page numbers]}
    """

    # First embed the question
    query_embedding = model.encode(question)

    # Retrieve relavant chunks
    results = vector_store.search(query_embedding,top_k = top_k)


    if not results: 
        return {"answer": "Sorry! Koi relevant information nahi mili document mein."}

    # Here we build context string from retrieved chunks
    context =  "\n\n".join([f"[Page {r['page']}] : {r['chunk']}" for r in results])


    # build prompt
    prompt = f"""You are a helpful assistant answering questions about a document.
            Use ONLY the context below to answer the question. If the answer isn't in the context, say you don't know.

            Context:
            {context}

            Question: {question}

            Answer:"""



    # call Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    sources = sorted(set(r["page"] for r in results))
    
    return {"answer": answer, "sources": sources}





if __name__ == "__main__":
    from utils.pdf_parser import extract_text_from_pdf
    from utils.embedding_engine  import chunk_text,embed_chunks
    from utils.vector_store  import VectorStore


    test_pdf_path = r"C:\Users\Adarsh Tiwari\OneDrive\Desktop\IIT IMP Documents\1st sem\Linear Algebra\Assignments\assignment 17.pdf"
    pages = extract_text_from_pdf(test_pdf_path)
    chunks = chunk_text(pages)
    chunks_with_embeddings = embed_chunks(chunks)


    store = VectorStore(embedding_dim=384)
    store.add_chunks(chunks_with_embeddings)


    text_question = "What does this document explain ?"     # Apna relevant question ya rakho

    result = answer_question(text_question,store)

    print("Answer:", result["answer"])
    print("Sources: Page(s)", result["sources"])