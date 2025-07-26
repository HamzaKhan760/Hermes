from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
import psycopg2
import os
from dotenv import load_dotenv
from typing import TypedDict
import re

# ------------------ Database Setup ------------------

load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')


# ------------------ Agent State ------------------

class AgentState(TypedDict):
    user_prompt: str
    schema_context: str
    llm_response: str
    final_output: str


# ------------------ Agent Controller ------------------

class AgentController:
    def __init__(self, use_cached_schema: bool = True):
        self.llm = Ollama(model="mistral")
        self.schema_context = None

        if use_cached_schema:
            self.schema_context = self.get_schema_description()

        self.agent_graph = self.build_graph()

    # -------- Database Utilities --------

    def get_schema_description(self) -> str:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
            cur = conn.cursor()

            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            rows = cur.fetchall()

            schema = {}
            for table, column, data_type in rows:
                if table not in schema:
                    schema[table] = []
                schema[table].append(f"{column} ({data_type})")

            cur.execute("""
                SELECT
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM 
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                ORDER BY source_table;
            """)
            fk_rows = cur.fetchall()

            cur.close()
            conn.close()

            schema_text = "Database Schema:\n"
            for table, columns in schema.items():
                schema_text += f"- {table}: " + ", ".join(columns) + "\n"

            if fk_rows:
                schema_text += "\nTable Relationships (Foreign Keys):\n"
                for source_table, source_column, target_table, target_column in fk_rows:
                    schema_text += f"- {source_table}.{source_column} references {target_table}.{target_column}\n"

            return schema_text

        except Exception as e:
            return f"Error fetching schema: {e}"

    def run_sql(self, query: str) -> str:
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

            if query.strip().lower().startswith("select"):
                rows = cur.fetchall()
                if rows:
                    # Get column names for better formatting
                    column_names = [desc[0] for desc in cur.description]
                    
                    # Format as a readable table
                    result = f"Query Results:\n"
                    result += " | ".join(column_names) + "\n"
                    result += "-" * (len(" | ".join(column_names))) + "\n"
                    
                    for row in rows:
                        result += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"
                else:
                    result = "No results found."
            else:
                conn.commit()
                result = "Query executed successfully."

            cur.close()
            conn.close()
            return result

        except Exception as e:
            return f"Error executing SQL: {e}"

    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response, handling markdown code blocks"""
        
        # First try to find Action Input pattern
        if "Action Input:" in response:
            start = response.find("Action Input:") + len("Action Input:")
            query_section = response[start:].strip()
        else:
            query_section = response.strip()
        
        # Remove markdown code blocks if present
        # Handle ```sql ... ``` pattern
        sql_block_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        sql_match = re.search(sql_block_pattern, query_section, re.DOTALL | re.IGNORECASE)
        
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no code blocks found, clean up the query section
        # Remove common prefixes and clean whitespace
        lines = query_section.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Action:') and not line.startswith('Thought:'):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()

    # -------- LangGraph Steps --------

    def schema_step(self, state: AgentState) -> AgentState:
        if self.schema_context:
            state["schema_context"] = self.schema_context
        else:
            state["schema_context"] = self.get_schema_description()
        return state

    def llm_step(self, state: AgentState) -> AgentState:
        if state["schema_context"].startswith("Error"):
            state["final_output"] = state["schema_context"]
            return state

        prompt = (
            f"{state['schema_context']}\n\n"
            "You are a SQL expert. Based on the database schema above, write a SQL query to answer the user's question.\n"
            "Respond with ONLY the SQL query, no explanations or markdown formatting.\n"
            "If you must use formatting, use this pattern:\n"
            "Action: SQLExecuter\n"
            "Action Input: YOUR_SQL_QUERY_HERE\n\n"
            f"User request: {state['user_prompt']}"
        )
        llm_output = self.llm.invoke(prompt)
        state["llm_response"] = llm_output
        return state

    def tool_step(self, state: AgentState) -> AgentState:
        llm_response = state["llm_response"]

        if "Action: SQLExecuter" in llm_response or "SELECT" in llm_response.upper():
            query = self.extract_sql_from_response(llm_response)
            
            if query:
                result = self.run_sql(query)
                state["final_output"] = result
            else:
                state["final_output"] = "Could not extract SQL query from response."
        else:
            state["final_output"] = llm_response

        return state

    # -------- Build Graph --------

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("SchemaFetcher", self.schema_step)
        graph.add_node("LLMResponder", self.llm_step)
        graph.add_node("ToolExecutor", self.tool_step)

        graph.set_entry_point("SchemaFetcher")
        graph.add_edge("SchemaFetcher", "LLMResponder")
        graph.add_edge("LLMResponder", "ToolExecutor")
        graph.add_edge("ToolExecutor", END)

        return graph.compile()

    # -------- Public Interface --------

    def handle_input(self, user_prompt: str) -> str:
        initial_state = {"user_prompt": user_prompt}
        final_state = self.agent_graph.invoke(initial_state)
        return final_state["final_output"]