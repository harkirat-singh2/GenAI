from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

def word_counter(text):
  return len(text.split())


prompt1 = PromptTemplate(
  template="Write a joke about a {topic}",
  input_variables=["topic"]
)

prompt2=PromptTemplate(
  template="Explain the joke {joke}",
  input_variables=["joke"]
)

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser= StrOutputParser()

joke_gen_chain= RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
  'joke':RunnablePassthrough(),
  'word_count':RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'cat'})


final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)