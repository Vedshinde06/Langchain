from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model_1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
gpt_model = ChatHuggingFace(llm=model_1)


class Feedback(BaseModel):
    
    sentiment: Literal['positive', 'negative'] = Field(description ='Give the sentiment of the feedback.')

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template='Give the sentiment of the following text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

classifier_chain = prompt | gpt_model | parser 

prompt2 = PromptTemplate(
    template='Give an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Give an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

parser1 = StrOutputParser()

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | gpt_model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | gpt_model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a crazy awesome phone'}))

chain.get_graph().print_ascii()