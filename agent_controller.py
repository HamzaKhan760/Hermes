from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import psycopg2
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama

#Database Setup
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

#Load Model
llama = Ollama(model="llama3")

#Get SQL Context
def get_schema_description():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()

        # Get table and column info
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        schema = {}
        for table, column, data_type in rows:
            if table not in schema:
                schema[table] = []
            schema[table].append(f"{column} ({data_type})")

        # Format schema nicely
        schema_text = "Database Schema:\n"
        for table, columns in schema.items():
            schema_text += f"- {table}: " + ", ".join(columns) + "\n"

        return schema_text

    except Exception as e:
        return f"Error fetching schema: {e}"

#SQL Query Tool
def run_sql(query):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(query)
        
        # Fetch results if it's a SELECT query
        if query.strip().lower().startswith("select"):
            rows = cur.fetchall()
            result = "\n".join(str(row) for row in rows)
        else:
            conn.commit()
            result = "Query executed successfully."

        cur.close()
        conn.close()
        return result

    except Exception as e:
        return f"Error executing SQL: {e}"

class AgentController:
    def __init__(self):
        self.llm = llama

        #Define tools
        self.tools = [
            Tool(
                name="SQLExecuter",
                func=run_sql,
                description="Executes SQL queries on the sales database and returns the results"
            )
        ]

        #Define Langchain Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose = True
        )

    def handle_input(self, user_prompt):
        schema_context = get_schema_description()

        if schema_context.startswith("Error"):
            return schema_context  # If schema fetch fails, return error

        prompt = (
            f"{schema_context}\n\n"
            "When deciding to use a tool, format your response strictly as:\n"
            "Action: [ToolName]\n"
            "Action Input: [Your input for the tool, SQL query or otherwise]\n"
            "Do NOT wrap the input inside parentheses or quotes on the Action line.\n\n"
            f"User request: {user_prompt}"
        )
        response = self.agent.run(prompt)
        return response
    