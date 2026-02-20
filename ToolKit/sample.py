from langchain_core.tools import tool

@tool
def add(a:int,b:int):
  """Adding two numbers"""
  return a+b

@tool
def multiply(a:int,b:int):
  """Multiplying two numbers"""
  return a*b

class MathToolkit:
  def get_tools(self):
    return[add,multiply]
  
toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
  print(tool.name, "=>", tool.description)  