import os
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt=PromptTemplate(
  template="Write a summary about the poem {poem}",
  input_variables=["poem"]
)

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()



loader = TextLoader('cricket.txt', encoding='utf8')

documents = loader.load()

print(documents[0].metadata)

chain= prompt | model | parser

result = chain.invoke(documents[0].page_content)

print(result)


