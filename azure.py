# azure.py
# Import necessary libraries
import streamlit as st
import openai
import os
import xml.etree.ElementTree as ET  # For parsing XML

# Set up your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize Streamlit with the title
st.title("MA Legal LLM")

# Parse the XML dataset and store it in a data structure
tree = ET.parse('usc40.xml')  # Use the path to your XML dataset
root = tree.getroot()

data = []  # Store your parsed data here

# Extract relevant data from the XML and populate the 'data' list
# You will need to implement the logic to extract and store the data from your specific XML format.

# Create a text input field for user queries
user_input = st.text_input("Ask a question:")

# Change the temperature
temperature = 0.3

# Send the user's query to OpenAI GPT-3 and use the dataset for reference
if user_input:
    # Query your dataset based on user_input
    # You can use your dataset and user_input to provide context for the GPT-3 query

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=150,
        temperature=temperature
    )

    st.write(response['choices'][0]['text'].strip())
