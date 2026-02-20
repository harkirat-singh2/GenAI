from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# LLM (not embeddings!)
llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools([multiply])

response = llm_with_tools.invoke(
    [HumanMessage(content="What is 3 multiplied by 5?")]
)

print(response)
