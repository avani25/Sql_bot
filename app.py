import streamlit as st
import os
from dotenv import load_dotenv
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import create_openai_tools_agent
from langchain.load.dump import dumps
import re
import pandas as pd
from sqlalchemy import create_engine

# Load OPENAI API key from .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# List of database URIs for the dropdown
# List of database URIs for the dropdown, with a placeholder
databases = [
 # Placeholder as the first item
    "course_stat_database",
    "student_course_features_2",
    "advising_database",
    "hackathon",
    "Chinook"
    # Add more database URIs as needed
]

llm_models = [
 # Placeholder as the first item
    "gpt-3.5-turbo",
    "gpt-4",
    # Add more database URIs as needed
]

# Setting up the layout using columns. Adjust the widths as needed.
col1, col2 = st.columns([1, 1])

with col1:
    selected_database = st.sidebar.selectbox('Select Database', databases, index = 0)

with col2:
    selected_llm_model = st.sidebar.selectbox('Select LLM Model', llm_models, index=0)

# Text area for user to insert the query
user_query = st.text_area("Insert your query here:", height=20)

# Initialize the database and toolkit globally for dynamic update
db = None
toolkit = None

# Function to update database connection
def update_database_connection(database, model):
    global db, toolkit, agent, agent_executor  
    db = SQLDatabase.from_uri("sqlite:///./" + database + ".db")

    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    template = """You are specifically designed to generate SQL queries related to student academic data. 
            If there are questions about applications or leads say NO... No application or lead data is available. 
            Whenever there's a query regarding the count of students, ensure that the chatbot uses the distinct keyword to provide accurate counts without duplication. 
            Whenever a question is asked about course or class then use class_id and when question is about section then use full_class_id. JUST USE FULL_CLASS_ID if SECTION is asked
             Never use full_class_id just use class_id...
            If asked to provide list or all students then just provide emplid.
            System: Use the following pieces of context to answer the users question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Attendance percent is calculated by attended_count divided by attendance_meet"""
            
    messages = [
        SystemMessage(content=template),
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessage(content= SQL_FUNCTIONS_SUFFIX),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    prompt = prompt.partial(**context)
    llm = ChatOpenAI(model_name=model, temperature=0)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
    return_intermediate_steps=True,
    )
    
def parse_sql(text):
    # Remove outer quotes and unescape characters
    pattern = r'[`"\\\\]'

    # Use re.sub to replace the matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)

    cleaned_text = cleaned_text.strip().strip('"').replace("\\\\n", " ")
    cleaned_text = cleaned_text.strip().strip('"').replace("\\", "")
    return cleaned_text

# Button to execute the process
if st.button('Generate SQL Query'):
    update_database_connection(selected_database, selected_llm_model)
    
    # Placeholder for logic to process the query and generate SQL.
    # This will likely involve calling the selected LLM model with the user query.
    # For demonstration, I'm just echoing back the user query in uppercase.
    api_return = agent_executor.invoke({"input": user_query})
    intermediate_steps = api_return["intermediate_steps"]
    
    sql_queries = [re.findall(r'"SELECT.*?"', str(action))[0] for action in intermediate_steps if re.findall(r'"SELECT.*?"', str(action))] 
    output = api_return["output"]
    generated_sql = parse_sql(str(sql_queries))  # Placeholder for actual SQL generation logic
    st.text_area("Output:", output, height=20, disabled=True)
    # Displaying the outputs in their respective boxes
    st.text_area("Generated SQL Query:", value=sql_queries, height=40, disabled=True)
    