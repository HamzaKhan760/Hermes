from models import OllamaLocal

llama = OllamaLocal()

class AgentController:
    def __init__(self):
        self.llm = llama

    def handle_input(self, user_prompt):
        instructions = (
        "You are Hermes, an AI reasoning agent designed to help users with sales data analysis.\n"
        "You can generate SQL queries, ask for clarification, or provide natural language answers based on the user's request.\n"
        "Decide what to do and explain your reasoning if necessary.\n\n"
        f"User Prompt: {user_prompt}\n"
        )

        llm_response = self.llm.generate_text(instructions)
        return llm_response