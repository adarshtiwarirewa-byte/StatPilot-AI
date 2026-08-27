from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a pdf file, page by page.

    Args : pdf_file path or pdf file like objext.

    Returns : list of dict : [{'page':1,"text" : text},{'page' : 2,"text":text},...,{..}]
    
    """
    reader = PdfReader(pdf_file)
    pages_content = []

    for page_num,page in enumerate(reader.pages,start = 1):
        text = page.extract_text()


        # For empty pages it will not run
        if text and text.strip():
            pages_content.append({"page" : page_num,"text":text.strip()})

    return pages_content



if __name__ == "__main__":
    # Standalone test
    test_pdf_path = r"C:\Users\Adarsh Tiwari\OneDrive\Desktop\IIT IMP Documents\1st sem\Linear Algebra\Assignments\assignment 17.pdf"  # apna test PDF ka path daalo
    result = extract_text_from_pdf(test_pdf_path)
    
    print(f"Total pages with text: {len(result)}")
    for page_data in result:
        print(f"\n--- Page {page_data['page']} ---")
        print(page_data['text'])  

