from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define structure
class Facts(BaseModel):
    facts: list[str] = Field(
        min_length=5,
        max_length=5,
        description="Exactly five facts"
    )

# Attach structure directly to model
structured_model = model.with_structured_output(Facts)

result = structured_model.invoke("Give five facts about Black Holes")

# Format output
for i, fact in enumerate(result.facts, 1):
    print(f"Fact {i}: {fact}")
