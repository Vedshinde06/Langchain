from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation')

llm = ChatHuggingFace(llm=model)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template = 'Write a poem on {topic}',
    input_variables=['topic']
)

poem_gen_chain = prompt | llm | parser

parallel_Chain = RunnableParallel({
    'poem': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = poem_gen_chain | parallel_Chain

result = final_chain.invoke({'topic': 'Love'})
  
result_final__boss = """{} \n word count - {}""".format(result['poem'], result['word_count'])
                                                              
print(result_final__boss)