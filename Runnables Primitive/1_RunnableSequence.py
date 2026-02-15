from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
  template="Write a joke about a {topic}",
  input_variables=["topic"]
)

prompt2=PromptTemplate(
  template="Explain the joke {joke}",
  input_variables=["joke"]
)

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser= StrOutputParser()

joke_chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(joke_chain.invoke({'topic':'cat'}))