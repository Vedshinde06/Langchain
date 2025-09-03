from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Write a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'generate a linkedin post on {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet' : prompt1 | llm | parser,
    'linkedin' : prompt2 | llm | parser
})

result = parallel_chain.invoke({'topic':'langchain'})

print(result['tweet'])
print(result['linkedin'])

