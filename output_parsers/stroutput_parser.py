from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace( llm=llm)

#1st prompt

template1 = PromptTemplate(
    template="Write a detailed report in {topic}",
    input_variables=['topic']
)

#2nd prompt

template1 = PromptTemplate(
    
)
