from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

chat = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct", task="text-generation")
model = ChatHuggingFace(
    llm=chat,
    temperature=0.7
)

chat_history = [
    SystemMessage(content="You are travel agent Perry. You are an expert in travel planning and can help users with their travel-related queries.")
]

while True:
    user_input = input("You:")
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Agent Perry:",response.content)
    if user_input.lower() == "exit":
        break
    
    