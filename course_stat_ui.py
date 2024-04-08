
import os
from dotenv import load_dotenv
import openai
import sqlite3
import tkinter as tk
import tkinter.ttk as ttk
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



# Connect to the database and execute the SQL script
# conn = sqlite3.connect('Kushal.db')
# with open('./Artist_Sqlite.sql', 'r',encoding='cp1252', errors='replace') as f:
#     sql_script = f.read()
# conn.executescript(sql_script)
# conn.close()

# OPEN AI key
load_dotenv()
import os
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# Create Executor
# db = SQLDatabase.from_uri("sqlite:///./student_course_features_2.db")
db = SQLDatabase.from_uri("sqlite:///./Avani.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
context = toolkit.get_context()
tools = toolkit.get_tools()
messages = [
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SQL_FUNCTIONS_SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)

# llm = ChatOpenAI( temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=toolkit.get_tools(),
#     verbose=True,
#     return_intermediate_steps=True,
# )

agent_executor = openai.Completion.create(
        engine="davinci",  # You can choose different engines such as "davinci" or "text-davinci"
        prompt=prompt,
        max_tokens=150,  # Maximum length of the completion
        temperature=0.7,  # Controls randomness; 0 is deterministic, 1 is maximum randomness
        stop="\n",  # Stop token to end the completion
        n=1  # Number of completions to generate
    )



# Create the agent executor
# db = SQLDatabase.from_uri("sqlite:///./Kushal.db")
# llm = OpenAI(temperature=0)
# toolkit = SQLDatabaseToolkit(db=db,  llm=llm)
# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True,
#     return_intermediate_steps=True
# )

# Create the UI window
root = tk.Tk()
root.title("Chat with your Course Stat Data")

# Create the text entry widget
entry = ttk.Entry(root, font=("Arial", 14))
entry.pack(padx=20, pady=10, fill=tk.X)

# Create the button callback
def on_click():
    # Get the query text from the entry widget
    query = entry.get()

    # Run the query using the agent executor
    api_return = agent_executor.invoke({"input": query})
    intermediate_steps = api_return["intermediate_steps"]
    sql_queries = [re.findall(r'`SELECT.*?`', str(action))[0] for action in intermediate_steps if re.findall(r'`SELECT.*?`', str(action))] 
    output = api_return["output"]
    # output = result+details
    # Display the result in the text widget
    text.delete("1.0", tk.END)
    text.insert(tk.END, output )

     # Display the result in the second text widget
    logic_text.delete("1.0", tk.END)
    logic_text.insert(tk.END, dumps(sql_queries, pretty=True))

# Create the button widget
button = ttk.Button(root, text="Chat", command=on_click)
button.pack(padx=20, pady=10)

# Create the text widget to display the result
text = tk.Text(root, height=5,  font=("Arial", 14))
text.pack(padx=20, pady=10)

# Create the second text widget to display the main output
logic_text = tk.Text(root, height=10,  font=("Arial", 14))
logic_text.pack(padx=20, pady=10)

# Start the UI event loop
root.mainloop()
