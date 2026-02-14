from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(
  template='Generate five interesting facts about {topic}',
  input_variables=["topic"]
)
  
# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

chain = prompt | model | parser

results = chain.invoke({'topic':'Black Holes'})

print(results)
print(type(results))

chain.get_graph().print_ascii()