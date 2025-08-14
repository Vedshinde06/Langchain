from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task= "text-generation")

chatmodel = ChatHuggingFace(llm=model)

prompt1 = PromptTemplate(
    
    template  = 'Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    
    template='Generate a one line summary on {topic}',
    input_variables=['summary']
)

parser = StrOutputParser()

chain = prompt1 | chatmodel | parser | prompt2 | chatmodel | parser 

result = chain.invoke({'topic':'Dr.Stone Anime'})

print(result)