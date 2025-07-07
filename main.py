# main.py
import streamlit as st
import os
from groq import Groq
from rag_deep import DocumentProcessor
import config

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Initialize Groq client and document processor
groq_client = Groq(api_key=config.GROQ_API_KEY)
doc_processor = DocumentProcessor()

# Streamlit UI setup (keep your existing CSS)
st.markdown("""
    <style>
    /* Your existing CSS styles */
    </style>
    """, unsafe_allow_html=True)

st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = PROMPT_TEMPLATE.format(
        user_query=user_query,
        document_context=context_text
    )
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",  
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = os.path.join("document_store/pdfs/", uploaded_pdf.name)
    doc_processor.save_uploaded_file(uploaded_pdf, saved_path)
    num_chunks = doc_processor.process_documents(saved_path)
    
    st.success(f"✅ Document processed successfully ({num_chunks} chunks indexed)! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = doc_processor.find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)