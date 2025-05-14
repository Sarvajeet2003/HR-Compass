from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

class DocumentSearch:
    def __init__(self, vector_db_path="faiss_index"):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the LLM for RAG"""
        # Use a smaller, open-access model instead
        model_id = "facebook/opt-350m"  # This is an open model that doesn't require login
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def create_vector_store(self, chunks):
        """Create vector store from document chunks"""
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_db_path)
    
    def load_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings)
            return True
        return False
    
    def setup_retrieval_qa(self):
        """Set up retrieval QA chain"""
        if not self.vector_store:
            if not self.load_vector_store():
                raise ValueError("Vector store not created yet")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
    
    def answer_question(self, question):
        """Answer question using RAG"""
        if not self.qa_chain:
            self.setup_retrieval_qa()
        
        result = self.qa_chain({"query": question})
        return result["result"]