import ollama
class OllamaLocal:
    def __init__(self, model = "llama3"):
        self.model = model
    
    def generate_text(self, prompt):
        response = ollama.chat(model=self.model, messages = [
        {"role": "user", "content": prompt}
        ])
        return response['message']['content']

#This file is no longer needed