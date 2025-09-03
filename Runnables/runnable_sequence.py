from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Write a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the following joke : {joke}',
    input_variables=['joke']
)

chain = RunnableSequence(prompt,llm,parser,prompt2,llm,parser)

print(chain.invoke({'topic':'One Piece'}))