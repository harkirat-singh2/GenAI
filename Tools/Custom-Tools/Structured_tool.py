from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
  a: int = Field(required=True, description="The first number to add")
  b: int = Field(required=True, description="The first number to add")

def multiply_func(a: int, b: int) ->int:
  return a * b

multiply_tool = StructuredTool.from_function(
  func=multiply_func,
  name="multiply",
  description="Multiply two numbers",
   args_schema=MultiplyInput  # Name of the Pydantic Class
)


result = multiply_tool.invoke({"a": 3, "b": 5})
print(result)
print(multiply_tool.name)
print(multiply_tool.args)
print(multiply_tool.description)