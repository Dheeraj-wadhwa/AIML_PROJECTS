import PyPDF2

def extract_text_from_pdf(file_stream):
    """
    Extract text content from a PDF file stream.
    
    Args:
        file_stream: A file-like object containing the PDF data
        
    Returns:
        str: The extracted text from the PDF
    """
    text = ""
    try:
        # Read the file stream using PyPDF2
        pdf_reader = PyPDF2.PdfReader(file_stream)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "
                
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    
    return text.strip()
