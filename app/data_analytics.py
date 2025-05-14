from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from utils.db_manager import DBManager
from utils.visualization import create_visualization
import google.generativeai as genai

class DataAnalytics:
    def __init__(self):
        self.db_manager = DBManager()
        self.llm_chain = None
        self.initialize_llm_chain()
    
    def initialize_llm_chain(self):
        """Initialize the LLM chain for text-to-SQL conversion using Gemini API"""
        # Set up the Gemini API key
        GEMINI_API_KEY = "AIzaSyCEd6f7y1XTUG4P42tEwSdT1_Nf-h76sRs"
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create a Gemini model instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=256,
            google_api_key=GEMINI_API_KEY
        )
        
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
        """Extract and clean SQL query from LLM output"""
        # Remove all code block markdown
        query = re.sub(r"```(?:sqlite|sql)?", "", text).strip("` \n")
        return query

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