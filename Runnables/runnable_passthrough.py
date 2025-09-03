from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'Generate a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the following joke : {joke}',
    input_variables='[joke]'
)

joke_gen_chain = prompt1 | llm | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': prompt2 | llm | parser
})

final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({'topic': 'indian politics'}))