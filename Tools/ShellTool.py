from langchain_community.tools import ShellTool

shellTool = ShellTool()

result = shellTool.invoke('whoami')

print(result)