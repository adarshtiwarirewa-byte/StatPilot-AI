from sentence_transformers import SentenceTransformer
from torch import embedding

model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_text(pages_content, chunk_size = 500,overlap=50):
    """
    Breaks page-wise text into overlapping chunks.
    
    Args:
        pages_content: list of dicts [{"page": 1, "text": "..."}, ...] from pdf_parser
        chunk_size: approx characters per chunk
        overlap: characters to overlap between consecutive chunks
    
    Returns:
        list of dicts: [{"page": 1, "chunk": "..."}, ...]
    """

    all_chunks = []
    for page_data in pages_content:
        page_num = page_data['page']
        text = page_data['text']

        start = 0
        while(start<len(text)):
            end = start + chunk_size
            chunk = text[start:end]

            all_chunks.append({"page":page_num,"chunk":chunk.strip()})

            start += chunk_size - overlap      # start next chunk with overlap of 50

    return all_chunks




def embed_chunks(chunks):
    """
    Converts list of text chunks into embeddings.
    
    Args:
        chunks: list of dicts [{"page": 1, "chunk": "..."}, ...]
    
    Returns:
        list of dicts with embeddings added: [{"page": 1, "chunk": "...", "embedding": [...]}, ...]
    """

    texts = [c["chunk"] for c in chunks]

    embeddings = model.encode(texts,show_progress_bar=True)

    for i,chunk_data in enumerate(chunks):
        chunk_data["embeddings"] = embeddings[i]

    return chunks


if __name__ == "__main__":
    # Standalone test
    from pdf_parser import extract_text_from_pdf
    
    test_pdf_path = r"C:\Users\Adarsh Tiwari\OneDrive\Desktop\IIT IMP Documents\1st sem\Linear Algebra\Assignments\assignment 17.pdf"  # apna test PDF ka path daalo
    pages = extract_text_from_pdf(test_pdf_path)
    
    chunks = chunk_text(pages)
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nSample chunk: {chunks[0]}")
    
    chunks_with_embeddings = embed_chunks(chunks)
    print(f"\nEmbedding shape for first chunk: {chunks_with_embeddings[0]['embeddings'].shape}")


