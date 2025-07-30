from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st 
from langchain.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task= "text-generation",
    temperature=1.5
)

model = ChatHuggingFace(llm=llm, verbose=True)

st.title("Research Paper Summarizer")

paper_input = st.selectbox("Select a research paper to summarize:", options=["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"])

style_input = st.selectbox("Select explanation style:", options=["Simple", "Technical", "Mathematical", "Fun"])

output_length = st.selectbox("Select output Length:", options = ["Short", "Medium", "Long"])

template = PromptTemplate(
    input_variables = ["paper", "style", "length"],
    template="""
    You are an expert in summarizing research papers.
    Summarize the {paper} research paper in a {style}.
    The summary should be in {length} length.
    The response directly start from the summary without any preamble like here is the summary or let me summarize it for you.
    You can make use of different fonts, bullet points, and other formatting to make the summary more readable.
    If you don't have information about the paper say so, do not make up the information.
    """)

if st.button("Summarize"):
        prompt = template.format(
            paper= paper_input,
            style= style_input,
            length= output_length
        )
        response = model.invoke(prompt)
        st.write(response.content)