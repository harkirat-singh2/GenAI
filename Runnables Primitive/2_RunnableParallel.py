from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt1=PromptTemplate(
  template="Write  Tweet  about a {topic}",
  input_variables=["topic"]
)

prompt2=PromptTemplate(
  template="Write  LinkedIn post  about a {topic}",
  input_variables=["topic"]
)

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model1,parser),
    'linkedin':RunnableSequence(prompt2,model1,parser)

  }
)

result = parallel_chain.invoke({'topic':'cat'})
print(result['tweet'])
print(result['linkedin'])