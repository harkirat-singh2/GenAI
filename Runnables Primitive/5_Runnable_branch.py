from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt1=PromptTemplate(
  template="Write a report about a {topic}",
  input_variables=["topic"]
)

prompt2 = PromptTemplate(
  template='Summarize the following text \n {text}',
  input_variables=['text']
)

parser= StrOutputParser()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

report_gen_chain=RunnableSequence(prompt1,model,parser)

branching_chain=RunnableBranch(
 (lambda x : len(x.split())>500,RunnableSequence(prompt2,model,parser)),RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branching_chain)

result = final_chain.invoke({'topic':'cat'})

print(result)