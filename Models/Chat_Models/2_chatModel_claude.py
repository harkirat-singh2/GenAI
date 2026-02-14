from langchain_anthropic import ChatAntropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAntropic(model='claude sonnet3.5')
print(model.invoke("What is the capital of Canada?"))
