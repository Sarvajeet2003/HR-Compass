import pandas as pd
import sqlite3
import os

class DBManager:
    def __init__(self, db_path="hr_data.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def csv_to_sqlite(self, csv_path, table_name):
        """Import CSV data into SQLite database"""
        df = pd.read_csv(csv_path)
        conn = self.connect()
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        self.close()
        return df.columns.tolist()
    
    def execute_query(self, query):
        """Execute SQL query and return results as DataFrame"""
        conn = self.connect()
        result = pd.read_sql_query(query, conn)
        self.close()
        return result
    
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