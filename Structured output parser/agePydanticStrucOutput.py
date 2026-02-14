from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
  template="""
  Generate the name , age AND CITY of a Ficional {place} person. \n
  {format_instructions}
  """
  ,
  input_variables=['place'],
  partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt =template.invoke({'place':'Indian'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)


chain = template | model | parser

final_result = chain.invoke({'place':'Indian'})

print(final_result)
print(type(final_result))