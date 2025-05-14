import PyPDF2
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
            return text
    
    def split_text_into_chunks(self, text, metadata=None):
        """Split text into semantic chunks"""
        chunks = self.text_splitter.create_documents([text], [metadata] if metadata else None)
        return chunks
    
    def process_pdf(self, pdf_path):
        """Process PDF file and return chunks with metadata"""
        filename = os.path.basename(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        metadata = {"source": filename}
        chunks = self.split_text_into_chunks(text, metadata)
        return chunks