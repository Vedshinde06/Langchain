from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(model='openai/gpt-oss-20b', task='text-generation')

model = ChatHuggingFace(llm=llm)

prompt = PromptTemlate(
    
)
