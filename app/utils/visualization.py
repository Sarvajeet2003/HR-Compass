import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

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
    
    plt.title(question)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic