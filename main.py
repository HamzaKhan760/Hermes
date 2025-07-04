from agent_controller import AgentController

print("Ask about your data:")

prompt = input("> ")

controller = AgentController()
response = controller.handle_input(prompt)

print(f"Hermes says:\n {response}")