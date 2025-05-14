import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_search import DocumentSearch
from data_analytics import DataAnalytics

# Initialize components
doc_search = DocumentSearch()
data_analytics = DataAnalytics()

# Set page config
st.set_page_config(
    page_title="HR Compass",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("HR Compass")
st.sidebar.info("Unified HR Knowledge and Analytics Platform")

# Create tabs
tab1, tab2 = st.tabs(["Document Search", "Analytics"])

# Document Search Tab
with tab1:
    st.header("HR Document Search")
    st.write("Ask questions about HR policies and procedures")
    
    doc_query = st.text_input("Enter your HR policy question:", placeholder="What are the US appraisal parameters for sales roles?")
    
    if st.button("Search Documents", key="doc_search"):
        if doc_query:
            with st.spinner("Searching documents..."):
                answer = doc_search.answer_question(doc_query)
                st.subheader("Answer")
                st.write(answer)
                st.info("The answer is based on the HR policy documents in the system.")
        else:
            st.warning("Please enter a question.")

# Analytics Tab
with tab2:
    st.header("HR Analytics")
    st.write("Ask questions about employee data and trends")
    
    analytics_query = st.text_input("Enter your analytics question:", placeholder="Show attrition rates by department in 2023")
    
    if st.button("Analyze Data", key="data_analytics"):
        if analytics_query:
            with st.spinner("Analyzing data..."):
                result = data_analytics.generate_visualization(analytics_query)
                
                st.subheader("SQL Query")
                st.code(result["sql_query"], language="sql")
                
                st.subheader("Results")
                st.dataframe(result["data"])
                
                st.subheader("Visualization")
                st.image(f"data:image/png;base64,{result['visualization']}")
        else:
            st.warning("Please enter a question.")