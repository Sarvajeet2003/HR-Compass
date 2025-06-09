# HR-Compass Technical Architecture Analysis

HR-Compass is a sophisticated HR management platform that combines document search capabilities with data analytics. Let me break down the technical implementation in detail:

## Core Technical Components

### 1. Document Search System (RAG Implementation)

The document search functionality is implemented using a Retrieval-Augmented Generation (RAG) approach:

- **Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` model to convert text into vector embeddings
- **Vector Store**: Implements FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Language Model**: Uses Facebook's OPT-350M model for generating answers
- **Document Processing**: 
  - Utilizes PyPDF2 for PDF text extraction
  - Implements `RecursiveCharacterTextSplitter` for chunking documents with configurable overlap

```python:/Users/sarvajeethuk/Desktop/HR-Compass/app/document_search.py
class DocumentSearch:
    def __init__(self, vector_db_path="faiss_index"):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
        self.initialize_llm()
    
    def answer_question(self, question):
        """Answer question using RAG"""
        if not self.qa_chain:
            self.setup_retrieval_qa()
        
        result = self.qa_chain({"query": question})
        return result["result"]
```

### 2. Data Analytics Engine

The analytics component translates natural language to SQL queries:

- **LLM for SQL Generation**: Uses Mistral-7B-Instruct-v0.2 with 4-bit quantization for efficiency
- **Database Management**: Custom SQLite implementation with schema introspection
- **Prompt Engineering**: Sophisticated prompt template that includes database schema information
- **Query Extraction**: Regex-based extraction of SQL from LLM output

```python:/Users/sarvajeethuk/Desktop/HR-Compass/app/data_analytics.py
def initialize_llm_chain(self):
    """Initialize the LLM chain for text-to-SQL conversion"""
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantization configuration for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # use nf4, NOT fp4, for CPU compatibility
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True  # this allows offloading to CPU in 32-bit
    )
    
    # ... existing code ...
```

### 3. Visualization Engine

The visualization component automatically selects appropriate chart types:

- **Chart Selection Logic**: Intelligently chooses visualization type based on query keywords and data structure
- **Chart Types**: Supports bar charts (comparisons), line charts (trends), and histograms (distributions)
- **Base64 Encoding**: Converts matplotlib plots to base64 strings for web display

```python:/Users/sarvajeethuk/Desktop/HR-Compass/app/utils/visualization.py
def create_visualization(df, question):
    """Create appropriate visualization based on data and question"""
    plt.figure(figsize=(10, 6))
    
    # Determine chart type based on data and question
    if "compare" in question.lower() or "comparison" in question.lower():
        if len(df.columns) >= 2:
            sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
    elif "trend" in question.lower() or "over time" in question.lower():
        if len(df.columns) >= 2:
            plt.plot(df[df.columns[0]], df[df.columns[1]], marker='o')
    elif "distribution" in question.lower():
        if len(df.columns) >= 1:
            sns.histplot(df[df.columns[1]], kde=True)
    else:
        # Default to bar chart for most queries
        if len(df.columns) >= 2:
            sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
    
    # ... existing code ...
```

### 4. Database Management

The database component handles data storage and retrieval:

- **SQLite Backend**: Uses SQLite for lightweight, file-based database storage
- **Pandas Integration**: Leverages pandas for data manipulation and SQL query execution
- **Schema Introspection**: Dynamically extracts database schema for LLM context

```python:/Users/sarvajeethuk/Desktop/HR-Compass/app/utils/db_manager.py
def get_schema_info(self):
    """Get database schema information for LLM context"""
    conn = self.connect()
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema_info[table_name] = [col[1] for col in columns]
    
    self.close()
    return schema_info
```

### 5. User Interface

The UI is built with Streamlit for rapid development:

- **Tab-based Interface**: Separates document search and analytics functionality
- **Interactive Elements**: Text inputs, buttons, and spinners for user interaction
- **Results Display**: Formatted output with code blocks, dataframes, and visualizations

## Technical Optimizations

1. **Memory Efficiency**:
   - Uses 4-bit quantization for the Mistral-7B model
   - Implements CPU offloading for handling larger models on limited hardware

2. **Performance Considerations**:
   - FAISS vector store for fast similarity search
   - Pre-processes documents and stores embeddings for quick retrieval
   - Uses smaller models where appropriate (OPT-350M for RAG)

3. **Modular Architecture**:
   - Clear separation of concerns between components
   - Well-defined interfaces between modules
   - Reusable utility classes for common operations

## Data Flow

1. **Document Search Flow**:
   - PDF documents → Text extraction → Chunking → Vector embeddings → FAISS index
   - User query → Vector embedding → Similarity search → Context retrieval → LLM generation → Answer

2. **Analytics Flow**:
   - User question → Schema-aware prompt → LLM → SQL generation → Query execution → Results → Visualization

## Dependencies

The project relies on several key libraries:

- **LangChain**: For RAG implementation and LLM chains
- **Hugging Face Transformers**: For accessing and running ML models
- **FAISS**: For vector similarity search
- **PyTorch**: For ML model operations
- **Streamlit**: For web UI
- **Pandas/Matplotlib/Seaborn**: For data manipulation and visualization

This architecture demonstrates a sophisticated application of modern NLP techniques to create a practical HR tool that combines document search with data analytics capabilities.

        Too many current requests. Your queue position is 5. Please wait for a while or switch to other models for a smoother experience.
