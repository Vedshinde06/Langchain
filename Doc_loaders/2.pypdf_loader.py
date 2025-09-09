from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Doc_loaders/TE_AI-DS_Syllabus_2022.pdf')

docs = loader.load()

print(docs[53].page_content)