from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=1.5,max_completion_tokens=10)

result = model.invoke("What is the capital of Canada?")

print(result)