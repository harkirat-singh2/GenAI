from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header("📚 Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select ...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
)

prompt=load_prompt('template.json')


# LCEL Chain (Modern Way)
chain = prompt | model | StrOutputParser()

if st.button("Summarize"):

    if paper_input == "Select ...":
        st.warning("Please select a research paper.")
    else:
        result = chain.invoke({
            "paper_name": paper_input,
            "style": style_input,
            "length": length_input
        })

        st.subheader("📄 Summary")
        st.write(result)

  
  
  
  
  
  
  
  
  
'''
  Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input} ]
1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.
  '''