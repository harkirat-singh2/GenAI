from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# 1️⃣ Define input schema
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")


# 2️⃣ Create custom tool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput  # MUST be inside class

    def _run(self, a: int, b: int) -> int:
        return a * b


# 3️⃣ Instantiate tool
multiply_tool = MultiplyTool()

# 4️⃣ Invoke
result = multiply_tool.invoke({"a": 3, "b": 3})

print("Result:", result)
print("Name:", multiply_tool.name)
print("Description:", multiply_tool.description)
print("Args Schema:", multiply_tool.args)
