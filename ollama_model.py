from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import Dict, List, Optional, Text

class OllamaModel:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)
        
    def model_(self, system_prompt: Optional[Text]) -> Dict:
        human_prompt = "{input}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        prompt = prompt | self.llm
        return prompt
    
    def raw_model(self):
        return self.llm