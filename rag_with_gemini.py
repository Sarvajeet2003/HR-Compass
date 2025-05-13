import os
import PyPDF2
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv # Add this import

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# API_KEY is now loaded from the .env file
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file or environment variables.")
    print("Please create a .env file in the project root with GEMINI_API_KEY=your_key")
    exit()

PDF_FILES = ["HR-Rules.pdf", "Rules.pdf"] # Files in your workspace
WORKSPACE_PATH = "/Users/sarvajeethuk/Desktop/HR-Compass" # Your workspace path

# Configure the Gemini API
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure your API key is correct and valid.")
    exit()

# --- 1. PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Add empty string if None
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# --- 2. Text Chunking ---
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if end >= len(text):
            break
    return [chunk for chunk in chunks if chunk.strip()] # Remove empty chunks

# --- 3. Generate Embeddings ---
# Using Gemini's embedding model
def get_embeddings(texts, model_name="models/embedding-001"):
    """Generates embeddings for a list of texts."""
    embeddings_list = []
    try:
        for i, text_batch in enumerate(batch_texts(texts, batch_size=100)): # Gemini API might have batch limits
            print(f"Processing batch {i+1} for embeddings...")
            # Filter out empty or whitespace-only strings before sending to API
            valid_texts = [t for t in text_batch if t and t.strip()]
            if not valid_texts:
                print(f"Skipping empty batch {i+1}")
                embeddings_list.extend([None] * len(text_batch)) # Placeholder for original batch size
                continue

            result = genai.embed_content(
                model=model_name,
                content=valid_texts,
                task_type="RETRIEVAL_DOCUMENT" # or "SEMANTIC_SIMILARITY" / "RETRIEVAL_QUERY"
            )
            # Align results back if some texts were filtered
            emb_idx = 0
            batch_embeddings = []
            for original_text in text_batch:
                if original_text and original_text.strip():
                    if emb_idx < len(result['embedding']):
                        batch_embeddings.append(result['embedding'][emb_idx])
                        emb_idx += 1
                    else: # Should not happen if API returns embeddings for all valid texts
                        batch_embeddings.append(None)
                else:
                    batch_embeddings.append(None) # No embedding for empty original text
            embeddings_list.extend(batch_embeddings)

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return None for all if batch fails, or handle per-item
        return [None] * len(texts)
    return embeddings_list

def batch_texts(texts, batch_size=100):
    """Yields successive N-sized chunks from texts."""
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

# --- 4. Simple In-Memory Vector Store & Retrieval ---
document_store = [] # Will store tuples of (text_chunk, embedding_vector)

def add_to_document_store(chunks, embeddings):
    global document_store
    for chunk, emb in zip(chunks, embeddings):
        if emb is not None and chunk.strip(): # Only add if embedding was successful and chunk is not empty
            document_store.append({"text": chunk, "embedding": np.array(emb)})

def retrieve_relevant_chunks(query_embedding, top_n=3):
    if not document_store:
        return []
    
    similarities = []
    for item in document_store:
        # Ensure item["embedding"] is a 2D array for cosine_similarity
        doc_emb_2d = np.array(item["embedding"]).reshape(1, -1)
        query_emb_2d = np.array(query_embedding).reshape(1, -1)
        sim = cosine_similarity(query_emb_2d, doc_emb_2d)[0][0]
        similarities.append({"text": item["text"], "score": sim})
    
    # Sort by score in descending order
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return [item["text"] for item in similarities[:top_n]]

# --- 5. Generate Answer with Gemini ---
def generate_answer(query, context_chunks, model_name="gemini-2.0-flash"): # Changed "gemini-pro" to "gemini-1.0-pro"
    prompt = "You are an assistant(chatbot). Answer the following question based *only* on the provided context. If the context doesn't contain the answer, say 'I don't have enough information from the provided documents to answer that.'\n\n"
    prompt += "Context from HR documents:\n"
    for i, chunk in enumerate(context_chunks):
        prompt += f"--- Context Chunk {i+1} ---\n{chunk}\n\n"
    prompt += f"--- User's Question ---\nQuestion: {query}\n\nAnswer:"

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "Sorry, I encountered an error while trying to generate an answer."

# --- Main RAG Pipeline Execution ---
def main():
    print("Starting RAG model setup...")

    # Load and process PDFs
    all_texts = []
    for pdf_file_name in PDF_FILES:
        pdf_path = os.path.join(WORKSPACE_PATH, pdf_file_name)
        print(f"Processing PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file {pdf_path} not found. Skipping.")
            continue
        text = extract_text_from_pdf(pdf_path)
        if text.strip(): # Check if text was actually extracted
            all_texts.append(text)
        else:
            print(f"Warning: No text extracted from {pdf_path}. Skipping.")

    if not all_texts:
        print("No text could be extracted from any PDF files. Exiting.")
        return

    full_document_text = "\n\n".join(all_texts)
    
    print("Chunking text...")
    chunks = chunk_text(full_document_text)
    if not chunks:
        print("No text chunks generated. This might be due to very short or empty PDF content. Exiting.")
        return
    print(f"Generated {len(chunks)} chunks.")

    print("Generating embeddings for chunks...")
    chunk_embeddings = get_embeddings(chunks)
    
    # Filter out chunks where embedding generation failed
    valid_chunks = []
    valid_embeddings = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        if emb is not None:
            valid_chunks.append(chunk)
            valid_embeddings.append(emb)
        else:
            print(f"Warning: Failed to generate embedding for a chunk. Skipping it.")

    if not valid_chunks:
        print("No valid embeddings could be generated for any chunks. Exiting.")
        return

    print(f"Successfully generated embeddings for {len(valid_chunks)} chunks.")
    add_to_document_store(valid_chunks, valid_embeddings)
    print(f"Document store populated with {len(document_store)} items.")

    # --- Example Query ---
    print("\nRAG Model is ready. You can now ask questions.")
    while True:
        user_query = input("Ask a question about the HR documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue

        print(f"\nProcessing query: '{user_query}'")
        
        query_embedding_result = get_embeddings([user_query], model_name="models/embedding-001") # Use RETRIEVAL_QUERY if supported and different
        
        if not query_embedding_result or query_embedding_result[0] is None:
            print("Could not generate an embedding for the query. Please try a different query.")
            continue
        
        query_embedding = query_embedding_result[0]

        print("Retrieving relevant context...")
        relevant_chunks = retrieve_relevant_chunks(query_embedding, top_n=3)

        if not relevant_chunks:
            print("No relevant context found in the documents for your query.")
            answer = "I don't have enough information from the provided documents to answer that."
        else:
            print(f"Found {len(relevant_chunks)} relevant context chunks.")
            print("Generating answer...")
            answer = generate_answer(user_query, relevant_chunks)
        
        print("\nAnswer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()