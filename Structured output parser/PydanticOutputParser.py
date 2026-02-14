from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define strict schema
class Facts(BaseModel):
    facts: list[str] = Field(
        description="A list of exactly five facts"
    )

parser = PydanticOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template="""
Generate exactly five facts about {topic}.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({"topic": "Black Holes"})

print(result)
print(type(result))
