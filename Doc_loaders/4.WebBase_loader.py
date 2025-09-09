from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

url = 'https://example.com'  # Simple test website

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html",
    "Accept-Encoding": "identity"  # Prevent GZIP compression
}

loader = WebBaseLoader(
    url,
    verify_ssl=True,  # Enable SSL verification
    requests_kwargs={"headers": headers}
)

docs = loader.load()

prompt = PromptTemplate(
    template = 'What do you see on the given url? \n{text}',
    input_variables=['text']
)

chain = prompt | llm | parser

chain.invoke({'text':docs[0].page_content})