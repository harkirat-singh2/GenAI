from langchain_community.tools import DuckDuckGoSearchRun

searchtool = DuckDuckGoSearchRun()

result = searchtool.invoke("AI Summit in India")

print(result)