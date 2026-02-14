from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(temperature=0.9)

print(llm.predict("What is the capital of Canada?"))