# yt_rag.py

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# ---------------------------------
# FETCH TRANSCRIPT (ENGLISH FIRST)
# ---------------------------------

def get_transcript(video_id):
    api = YouTubeTranscriptApi()

    print("Fetching transcript...")

    try:
        # Try English first
        transcript_list = api.fetch(video_id, languages=["en"])
        print("Using English transcript.")
    except NoTranscriptFound:
        print("English not available. Using first available language...")
        transcript_list = api.fetch(video_id)

    full_text = " ".join(chunk.text for chunk in transcript_list)
    return full_text


# ---------------------------------
#  SPLIT TEXT INTO CHUNKS
# ---------------------------------

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]


# ---------------------------------
#  CREATE VECTOR STORE
# ---------------------------------

def create_vectorstore(docs):
    print("Creating embeddings and vector store...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embedding_model)


# ---------------------------------
#  LOAD LIGHTWEIGHT LLM (CPU SAFE)
# ---------------------------------

def load_llm():
    print("Loading TinyLlama model (CPU mode)...")

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.3,
        device=-1
    )

    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------------
#  BUILD CHAIN
# ---------------------------------

def build_chain(llm):
    template = """
You are a helpful assistant.

Answer ONLY from the provided context.
Respond in clear English.
Keep the answer short (3-4 sentences).
Do not add extra suggestions.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    return prompt | llm | parser


# ---------------------------------
#  ASK QUESTION
# ---------------------------------

def ask_question(vectorstore, llm, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = build_chain(llm)

    return chain.invoke({
        "context": context,
        "question": question
    })


# ---------------------------------
# MAIN PROGRAM
# ---------------------------------

if __name__ == "__main__":

    video_url = input("Enter YouTube URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]

    transcript = get_transcript(video_id)

    print("Splitting transcript...")
    docs = split_text(transcript)

    vectorstore = create_vectorstore(docs)

    llm = load_llm()

    print("\n✅ Ready! Ask questions about the video.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your Question: ")

        if question.lower() == "exit":
            break

        answer = ask_question(vectorstore, llm, question)

        print("\nAnswer:\n")
        print(answer)
        print("-" * 60)
