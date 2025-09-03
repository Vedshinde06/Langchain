from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Write a report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a summary of the following text : {text}',
    input_variables= ['text']
)

chain1 = prompt1 | llm | parser

chain2 = RunnableBranch(
    (lambda x : len(x.split())>250, prompt2 | llm | parser),
    RunnablePassthrough()
)

final_chain = chain1 | chain2

print(final_chain.invoke({'topic': 'Brocode'}))