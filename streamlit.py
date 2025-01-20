from typing import List
from langchain.schema import Document
from typing_extensions import TypedDict
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from vectorstore import vectorstore_output
from text_to_sql import database_response, get_sql_query_v2, fetch_data_from_db, get_mysql_conn, get_prompts
from get_model import llm_router_invoke
import streamlit as st

st.title("Ask questions to your mysql database and vectorstore")
llm=llm_router_invoke()
### Router
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field



# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "database"] = Field(
        ...,
        description="Given a user question choose to route it to database or a vectorstore.",
    )



structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or database.
The vectorstore contains documents related to SAS programming language and the database has information about different kind of IT assets.
Use the vectorstore for questions on these topics. Otherwise, use database."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "How many assets are there?"}
    )
)
#print(question_router.invoke({"question": "How many assets are there?"}))


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]



rag_chain= vectorstore_output()

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = rag_chain.invoke(question)
    return {"documents": documents, "question": question}


answer_chain= database_response()
db= get_mysql_conn()
prompt= get_prompts()

def database(state):
    """
    database search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with database results.
    """

    print("---database---")
    print("---HELLO--")
    question = state["question"]
    print(question)
    sql_query = get_sql_query_v2(prompt=prompt, db=db, question= question)
    sql_data = fetch_data_from_db(db=db, sql_query=sql_query)
    response = answer_chain.invoke(input={"question": question, "query": sql_query, "result": sql_data})

    #return {"documents": docs, "question": question}
    return {"documents": response, "question": question}

def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "database":
        print("---ROUTE QUESTION TO database SEARCH---")
        return "database"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    


workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("database", database)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "database": "database",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "database", END)
# Compile
app = workflow.compile()

question =st.chat_input()

inputs = {
    "question": question  # What is SAS programming language? # Give no of tests which had status as normal.
}
for output in app.stream(inputs):
    for key, value in output.items():
        #pprint(f"Node '{value}':")
    #pprint("\n---\n")
        data= f"Node '{value}':"

start = data.find("documents': '") + len("documents': '")
end = data.find("',", start)
extracted_text = data[start:end]
st.write(extracted_text)