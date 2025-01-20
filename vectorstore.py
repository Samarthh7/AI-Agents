from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from get_model import llm_vectorstore_invoke

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vector_store = QdrantVectorStore(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="SAS book",
    embedding=embedding_model,
)
prompt = ChatPromptTemplate.from_template("""
You are an expert in SAS programming language
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
If you dont know the answer say that I don't know.
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

system_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
"""
human_template = """
Question: {question}
Context: {context}
"""
system_message = SystemMessagePromptTemplate.from_template(template=system_template)
human_message = HumanMessagePromptTemplate.from_template(template=human_template)
prompt_template = ChatPromptTemplate([
    system_message, human_message
])

llm =llm_vectorstore_invoke()

def vectorstore_output():
    rag_chain = (
        {"context": retriever , "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain