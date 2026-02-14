from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chatTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

# load chat history
with open("chat_history.txt", "r") as f:
    chat_history = f.readlines()

print(chat_history)

# create prompt
prompt = chatTemplate.invoke({
    "chat_history": chat_history,
    "query": "Where is my order?"
})

print(prompt)
