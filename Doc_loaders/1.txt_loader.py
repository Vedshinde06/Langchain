from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Write a summary on {poem}',
    input_variables=['poem']
)

loader = TextLoader("Doc_loaders/cricket.txt", encoding="utf-8")

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | llm | parser

chain.invoke({'poem':docs[0].page_content})