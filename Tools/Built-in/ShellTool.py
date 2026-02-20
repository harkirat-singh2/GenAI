from langchain_community.tools import ShellTool

shellTool = ShellTool()

result = shellTool.invoke('whoami')

print(result)

print(shellTool.name)

print(shellTool.args)

print(shellTool.description)