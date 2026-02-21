from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage
import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------- TOOL 1 ----------------
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch currency conversion factor between base and target currency"""
    url = f"https://v6.exchangerate-api.com/v6/d4f20dc3ebb0a67d4a6dadf3/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    return data["conversion_rate"]


# ---------------- TOOL 2 ----------------
@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """Multiply base value with conversion rate"""
    return base_currency_value * conversion_rate


# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [
    HumanMessage(
        content="What is the conversion factor between USD and INR, and based on that can you convert 10 USD to INR?"
    )
]

# First LLM call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

conversion_rate = None

# Execute tool calls manually
for tool_call in ai_message.tool_calls:

    # ---------------- CALL TOOL 1 ----------------
    if tool_call["name"] == "get_conversion_factor":
        result = get_conversion_factor.invoke(tool_call["args"])
        conversion_rate = result

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # ---------------- CALL TOOL 2 ----------------
    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        result = convert.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

# Final LLM response
final_response = llm_with_tools.invoke(messages)

for block in final_response.content:
    if block["type"] == "text":
        print(block["text"])