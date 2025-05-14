from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai

class DocumentSearch:
    def __init__(self, vector_db_path="faiss_index"):
        self.vector_db_path = vector_db_path
        # Specify CPU device explicitly to avoid CUDA/MPS issues
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vector_store = None
        self.qa_chain = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the LLM for RAG using Gemini API"""
        # Set up the Gemini API key
        GEMINI_API_KEY = "AIzaSyCEd6f7y1XTUG4P42tEwSdT1_Nf-h76sRs"
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create a Gemini model instance - using gemini-pro instead of gemini-2.0-flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=256,
            google_api_key=GEMINI_API_KEY
        )
    
    def create_vector_store(self, chunks):
        """Create vector store from document chunks"""
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_db_path)
    
    def load_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.vector_db_path):
            try:
                self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        return False
    
    def setup_retrieval_qa(self):
        """Set up retrieval QA chain"""
        if not self.vector_store:
            if not self.load_vector_store():
                raise ValueError("Vector store not created yet")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 2})  # Reduced from 3
        )
    
    def answer_question(self, question):
        """Answer question using RAG"""
        if not self.qa_chain:
            self.setup_retrieval_qa()
        
        result = self.qa_chain({"query": question})
        return result["result"]