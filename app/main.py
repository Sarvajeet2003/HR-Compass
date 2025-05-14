import os
import argparse
from utils.pdf_processor import PDFProcessor
from utils.db_manager import DBManager
from document_search import DocumentSearch

def setup_document_search(pdf_dir):
    """Set up document search by processing PDFs and creating vector store"""
    pdf_processor = PDFProcessor()
    doc_search = DocumentSearch()
    
    all_chunks = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            chunks = pdf_processor.process_pdf(pdf_path)
            all_chunks.extend(chunks)
    
    print(f"Creating vector store with {len(all_chunks)} chunks...")
    doc_search.create_vector_store(all_chunks)
    print("Vector store created successfully!")

def setup_database(csv_path):
    """Set up database by importing CSV data"""
    db_manager = DBManager()
    print(f"Importing {csv_path} into database...")
    columns = db_manager.csv_to_sqlite(csv_path, "employees")
    print(f"CSV data imported successfully with columns: {columns}")

def main():
    parser = argparse.ArgumentParser(description="HR Compass Setup")
    parser.add_argument("--pdf_dir", default="./data/raw", help="Directory containing PDF files")
    parser.add_argument("--csv_path", default="./employee_data.csv", help="Path to employee CSV data")
    parser.add_argument("--setup_docs", action="store_true", help="Set up document search")
    parser.add_argument("--setup_db", action="store_true", help="Set up database")
    
    args = parser.parse_args()
    
    if args.setup_docs:
        setup_document_search(args.pdf_dir)
    
    if args.setup_db:
        setup_database(args.csv_path)
    
    if not args.setup_docs and not args.setup_db:
        print("No actions specified. Use --setup_docs or --setup_db")

if __name__ == "__main__":
    main()