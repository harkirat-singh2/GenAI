from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,ResponseSchema
from dotenv import load_dotenv

load_dotenv()

# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1️⃣ Detailed prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2️⃣ Summary prompt
template2 = PromptTemplate(
    template="Write a summary of the following text:\n{text}",
    input_variables=["text"]
)



parser =StrOutputParser()

chain = template1 |model |parser |template2 |model | parser 

result = chain.invoke({"topic":"Black Hole"})

print(result)