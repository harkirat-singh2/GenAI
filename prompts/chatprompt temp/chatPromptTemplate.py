from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template=([
  ('system','You are a helpful{domain} assistant'),
  ('human','Explain in simple terms what is{topic}')
])

prompt=chat_template.invoke({'domain':'Cricketer','topic':'Sachin Tendulkar'})