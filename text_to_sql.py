from langchain_community.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from get_model import llm_database_invoke

def get_mysql_conn():
    mysql_uri = "mysql+mysqlconnector://root:Pratap%408512@localhost:3306/medical_details"
    db = SQLDatabase.from_uri(mysql_uri)
    return db

examples = [
    {
        "input": "How many employees are there in the database?",
        "query": "SELECT COUNT(*) FROM employees;"
    },
    {
        "input": "Find all employees from the 'Marketing' department",
        "query": "SELECT first_name, last_name, email, phone_number, hire_date, job_id, salary, manager_id, departments.department_name as DepartmentName FROM employees JOIN departments ON employees.department_id = departments.department_id WHERE departments.department_name = 'Marketing';"
    },
    {
        "input": "Can you write a query to show department names and the average salary of employees in each department?",
        "query": """SELECT departments.department_name, AVG(employees.salary) as AverageSalary
                    FROM employees
                    JOIN departments ON employees.department_id = departments.department_id
                    GROUP BY departments.department_name;"""
    },
    {
        "input": "Which employee has the highest salary in the company?",
        "query": "SELECT employees.first_name, employees.last_name, MAX(employees.salary) as MaxSalary FROM employees GROUP BY employees.first_name, employees.last_name;"
    }
]

example_prompt = PromptTemplate(
    input_variables=["input", "query"], template="Question: {input}\n{query}"
)

def get_prompts():
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    Don't use ''' before the starting and ending of sql query.

    Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    \nBelow are a number of examples of questions and their corresponding SQL queries.
    """,
        suffix="""For the below question, PROVIDE JUST THE SQL QUERY AND DO NOT PROVIDE ANY EXPLAINATION OR SUCH.
        Question: {question}\nSQL Query:""",
        input_variables=["schema", "question"],
    )
    return prompt
def get_sql_query_v2(prompt: PromptTemplate, db: SQLDatabase, question: str):
    """
        Converts the user question to sql query.
    """
    # Get the schema of the database
    schema = db.get_context()['table_info']

    # Generate the response from the LLM
    chain: RunnableSequence = prompt | llm
    response = chain.invoke(input={'schema': schema, 'question': question})
    # print(f"LLM Response: \n{response}\n\n")

    # Extract the sql query from the response
    # sql_query = extract_sql_query(llm_response=response)

    return response


def fetch_data_from_db(db: SQLDatabase, sql_query: str):
    """
        Fetches the data by running the sql query on the database
    """
    data = db.run(sql_query)
    return data

llm =llm_database_invoke()

def database_response():
    template = """
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Return the answer in the markdown format.
    Answer: """

    answer_prompt = PromptTemplate.from_template(template=template)

    answer_chain: RunnableSequence = answer_prompt | llm | StrOutputParser()
    return answer_chain

# Getting the sql query
#sql_query = get_sql_query_v2(prompt=prompt, db=db, question=user_question)

# Fetching the data from the databse using sql query
# response=sql_query
# response

# ans=response.content
# ans

#sql_data = fetch_data_from_db(db=db, sql_query=ans)

# Final response
#     response = answer_chain.invoke(input={"question": user_question, "query": sql_query, "result": sql_data})
# print(response)