# azure.py
# Import necessary libraries
import streamlit as st
import openai
import os
# azure.py
# Import necessary libraries
import streamlit as st
import nltk
nltk.download('punkt')


# Replace these imports with your actual backend code
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set up your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize Streamlit
st.title("LLM Chatbot")

# Create a text input field for user queries
user_input = st.text_input("Ask a question:")

# Your backend code starts here

if user_input:
    # Your existing backend code
    query = user_input  # Assuming the user input is the query
    
    # Replace this block with your existing backend code
    dataset_corpus_path = "Short Boston Code.pdf"
    
    pdf_loader = PyPDFDirectoryLoader(dataset_corpus_path)
    documents = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100
    )
    
    chunks = pdf_loader.load_and_split(text_splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(chunks, embeddings)
    
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type="stuff")
    
    docs = db.similarity_search(query, k=2)
    
    result = chain.run(input_documents=docs, question=query)
    
    # Display the result
    st.write(result)  # Modify this to display the result as needed
