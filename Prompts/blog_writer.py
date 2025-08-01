from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()

chat = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct", task="text-generation")
model = ChatHuggingFace(
    llm=chat,
    temperature=1.2
)

prompt = PromptTemplate(
    input_vars=["topic"],
    
    template="""Generate a blog on {topic}. Include a catchy title, outline and a funfact for the blog. Output should be in Markdown."""
)

st.title("Blog Writer")

topic = st.text_input("Enter the topic for your blog:")

p1 = prompt.format(topic=topic)

if st.button("Lets Go"):
    response = model.invoke(p1)
    st.write(response.content)

#print(response.content)