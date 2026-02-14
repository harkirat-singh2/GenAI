from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = JsonOutputParser()


template=PromptTemplate(
    template="Write five facts about{topic}. \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result=parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({})



print(final_result) 
print(type(final_result))