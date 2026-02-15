from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt=PromptTemplate(
  template="Write a answer {question} to the following questions {text}",
  input_variables=["question","text"]
)

url = "https://www.flipkart.com/apple-macbook-air-m4-16-gb-256-gb-ssd-macos-sequoia-mc6t4hn-a/p/itm7c1831ce25509?pid=COMH9ZWQCJGMZGXE&lid=LSTCOMH9ZWQCJGMZGXEBSSIQU&marketplace=FLIPKART&cmpid=content_computer_22927808323_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,770553264708,,,,c,,,,,,,&entryMethod=22927808323&&cmpid=content_22927808323_gmc_pla&gad_source=1&gad_campaignid=22927808323&gbraid=0AAAAADxRY5_LDJx6LQlNxtygygihgLVrp&gclid=CjwKCAiA-sXMBhAOEiwAGGw6LEfs4Y4ce8rKwHaZfxlu5sAxaC0HrCSF81IDC7bM8v6fHUH21MXBchoCGMAQAvD_BwE"

loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

chain= prompt | model | parser

result = chain.invoke({'question':'What is the name of the product?','text':docs[0].page_content})

print(result)

