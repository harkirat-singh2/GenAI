from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

loader = PyPDFLoader('dl-curriculum.pdf')

docs=loader.load()

print(docs[0].page_content)





load_dotenv()

prompt=PromptTemplate(
  template="Write a summary about the poem {poem}",
  input_variables=["poem"]
)