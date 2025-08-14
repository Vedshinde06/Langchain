from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm  = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'Generate 2 interesting facts about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({'topic' : 'Science'})

print(result)

chain.get_graph().print_ascii()