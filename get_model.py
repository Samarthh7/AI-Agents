from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
def llm_router_invoke():
    llm_router=ChatGroq(groq_api_key="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",model_name="Gemma2-9b-It")
    return llm_router

def llm_database_invoke():
    llm_database = Ollama(base_url="http://10.75.22.61:11434",model="gemma2:27b")
    return llm_database

def llm_vectorstore_invoke():
    llm_vectorstore = Ollama(base_url="http://10.75.22.61:11434",model="llama3.1:8b")
    return llm_vectorstore
