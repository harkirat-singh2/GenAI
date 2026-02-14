from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create model
llm = HuggingFaceEndpoint(
repo_id="google/gemma-2-2b-it",
task="text-generation"

)

model = ChatHuggingFace(llm=llm)
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

# Generate detailed report
prompt1 = template1.format(topic="Black Hole")
result1 = model.invoke(prompt1)

# Generate summary
prompt2 = template2.format(text=result1)
result2 = model.invoke(prompt2)

print(result2)
