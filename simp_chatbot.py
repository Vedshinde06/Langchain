from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

chat = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
    temperature=0.8
)

model = ChatHuggingFace(llm=chat)

chat_history = [
    SystemMessage(content="You are a Musician.")
]

while True:
    user_input = input("You:")
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input=="Exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)