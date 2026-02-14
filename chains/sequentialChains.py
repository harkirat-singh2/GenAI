from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1=PromptTemplate(
  template='Generate detailed report about {topic}',
  input_variables=["topic"]
)
  
prompt2=PromptTemplate(
  template='Generate a five pointer summary from the following {text}',
  input_variables=["text"]
  
)
  
# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

results = chain.invoke({'topic':'Black Holes'})

print(results)
print(type(results))

