import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

deepseek_key = os.getenv("sk-baa93334c09b418fbc2ce94636c04231")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Use DeepSeek instead of OpenAI
llm = ChatOpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

llm_with_tools = llm.bind_tools([multiply])

response = llm_with_tools.invoke(
    [HumanMessage(content="What is 3 multiplied by 5?")]
)

print(response)