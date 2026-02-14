from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------- Structured Classification ----------

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

classification_prompt = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into
'positive' or 'negative'.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

classifier_chain = classification_prompt | model | parser

# ---------- Response Generators ----------

response_parser = StrOutputParser()

positive_prompt = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Convert Feedback object back to dictionary format
extract_feedback = RunnableLambda(lambda x: {"feedback": feedback})

# ---------- Branch Logic ----------

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive",
     extract_feedback | positive_prompt | model | response_parser),

    (lambda x: x.sentiment == "negative",
     extract_feedback | negative_prompt | model | response_parser),

    RunnableLambda(lambda x: "Could not determine sentiment")
)

# ---------- Final Pipeline ----------

chain = classifier_chain | branch_chain

feedback = "I love the new design of your website! It's so user-friendly and visually appealing."

result = chain.invoke({"feedback": feedback})

print(result)
print(type(result))

chain.get_graph().print_ascii()
