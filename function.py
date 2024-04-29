from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
embeddings = OpenAIEmbeddings()
chat_model = ChatOpenAI()
str_parser = StrOutputParser()
template = (
    "請根據以下內容加上自身判斷回答問題:\n"
    "{context}\n"
    "問題: {question}"
    )
prompt = ChatPromptTemplate.from_template(template)

def pdf_load(file_path):
  loader = PyPDFLoader(file_path=file_path)
  docs = loader.load()
  return docs

def office_file(file_path):
  loader = UnstructuredFileLoader(file_path)
  docs = loader.load()
  return docs

def splitter(docs, separators, chunk_size, chunk_overlap):
  text_splitter = RecursiveCharacterTextSplitter(
      separators=separators,
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap)
  splits = text_splitter.split_documents(docs)
  return splits

def rag(splits):
  db = FAISS.from_documents(splits, embeddings)
  db.save_local(folder_path="office_db")
  new_db = FAISS.load_local(
      folder_path="office_db",
      embeddings=embeddings,
      allow_dangerous_deserialization=True)
  retriever=new_db.as_retriever()
  chain = (
      {"context": retriever, "question": RunnablePassthrough()}
      | prompt
      | chat_model
      | str_parser
  )
  return chain
def pandas_agent(path, skiprows):
  df = pd.read_csv(path,skiprows=skiprows)
  agent = create_pandas_dataframe_agent(llm=chat_model,
                                        df=df,
                                        prefix='回答請使用繁體中文',
                                        agent_type="openai-tools")
  return agent
