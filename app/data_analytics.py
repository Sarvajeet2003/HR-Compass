from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import re
from utils.db_manager import DBManager
from utils.visualization import create_visualization

class DataAnalytics:
    def __init__(self):
        self.db_manager = DBManager()
        self.llm_chain = None
        self.initialize_llm_chain()
    
    def initialize_llm_chain(self):
        """Initialize the LLM chain for text-to-SQL conversion"""
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            llm_int8_enable_fp32_cpu_offload=True
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # use nf4, NOT fp4, for CPU compatibility
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # this allows offloading to CPU in 32-bit
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create prompt template for SQL generation
        schema_info = self.db_manager.get_schema_info()
        schema_str = self._format_schema_info(schema_info)
        
        template = """
        You are an expert SQL query generator. Given the following database schema and a question, 
        generate a valid SQL query that answers the question.
        
        Database Schema:
        {schema}
        
        User Question: {question}
        
        Generate only the SQL query without any explanation. The query should be valid SQLite syntax.
        SQL Query:
        """
        
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=template
        )
        
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    def _format_schema_info(self, schema_info):
        """Format schema information for prompt"""
        schema_str = ""
        for table, columns in schema_info.items():
            schema_str += f"Table: {table}\n"
            schema_str += f"Columns: {', '.join(columns)}\n\n"
        return schema_str
    
    def _extract_sql_query(self, text):
        """Extract SQL query from LLM output"""
        # Clean up the output to get just the SQL query
        query = text.strip()
        if "```sql" in query:
            query = re.search(r"```sql\n(.*?)```", query, re.DOTALL).group(1)
        elif "```" in query:
            query = re.search(r"```\n(.*?)```", query, re.DOTALL).group(1)
        return query.strip()
    
    def natural_language_to_sql(self, question):
        """Convert natural language question to SQL query"""
        schema_info = self.db_manager.get_schema_info()
        schema_str = self._format_schema_info(schema_info)
        
        result = self.llm_chain.run(schema=schema_str, question=question)
        sql_query = self._extract_sql_query(result)
        return sql_query
    
    def execute_query(self, question):
        """Execute SQL query generated from natural language question"""
        sql_query = self.natural_language_to_sql(question)
        result = self.db_manager.execute_query(sql_query)
        return {
            "sql_query": sql_query,
            "result": result
        }
    
    def generate_visualization(self, question):
        """Generate visualization from query results"""
        query_result = self.execute_query(question)
        df = query_result["result"]
        sql_query = query_result["sql_query"]
        
        visualization = create_visualization(df, question)
        return {
            "sql_query": sql_query,
            "visualization": visualization,
            "data": df
        }