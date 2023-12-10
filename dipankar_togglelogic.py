# azure.py
# Import necessary libraries
import streamlit as st
import openai
import os

import streamlit as st
import tiktoken


# Replace these imports with your actual backend code
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set up your OpenAI API key
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Initialize Streamlit
st.title("Boston City Code Chatbot")
st.subheader("Welcome to the Boston City Code Chatbot! Your one stop shop for all things Boston law.")
# Toggle switch to select mode
mode = st.radio("Select Mode:", ("Expert", "Novice"))

# User input field
user_input = st.text_input("Please input your question below:")

if user_input:
    # Process user query and generate response based on selected mode
    result = generate_response(user_input, mode)
    st.write("Response:", result)

# Function to generate responses based on mode
def generate_response(query, mode):
    dataset_corpus_path = "Short Boston Code.pdf"
    
    pdf_loader = PyPDFDirectoryLoader(dataset_corpus_path)
    documents = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100
    )
    
    chunks = pdf_loader.load_and_split(text_splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, temperature=0.3)
    db = FAISS.from_documents(chunks, embeddings)
    
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type="stuff")
    
    docs = db.similarity_search(query, k=2)
    
    result = chain.run(input_documents=docs, question=query)
    
    if mode == 'Expert':
        # Generate response in formal legal tone
        response = expert_prompt.generate_prompt(input=query).get_template()
    else:
        # Generate response in simpler language
        response = novice_prompt.generate_prompt(input=query).get_template()
    return response

from langchain.prompts import PromptTemplate

expert_template = """You are a lawyer for the City of Boston and a legal expert.
You are great at answering questions about Boston Municipal codes using leagl jargons.
However, you don't have the ability to answer in an easy to understand language.

Here is a question:
{input}"""

expert_prompt = PromptTemplate.from_template(expert_template)

novice_template = """You are a lawyer for the City of Boston and a legal expert.
You have a special talent for answering questions in a very easy language so that even laypersons without a background in law can understand the responses.
You never use complex legal words or jargon to give answer to questions.

Here is a question:
{input}"""

novice_prompt = PromptTemplate.from_template(novice_template)

routes = [
    {
        "name": "expert",
        "description": "Give expert responses using complex legal jargon if the mode selected is Expert",
        "prompt_template": expert_template
    },
    {
        "name": "novice",
        "description": "Give legal responses in a very easy to understand language if the mode selected is Novice",
        "prompt_template": novice_template
    }
]

llm = OpenAI()

from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain import PromptTemplate

destination_chains = {}
for p_info in routes:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{p['name']}: {p['description']}" for p in routes]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
