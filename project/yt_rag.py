# yt_rag.py

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# -------------------------------
# 1️⃣ GET TRANSCRIPT
# -------------------------------
def get_transcript(video_id):
    api = YouTubeTranscriptApi()

    try:
        # Try English first
        transcript_list = api.fetch(video_id, languages=["en"])
        print("Using English transcript.")
    except:
        # If English not available, use any available transcript
        print("English not available. Using available language transcript...")
        transcript_list = api.fetch(video_id)

    full_text = " ".join(chunk.text for chunk in transcript_list)
    return full_text




# -------------------------------
# 2️⃣ SPLIT TEXT INTO CHUNKS
# -------------------------------

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]


# -------------------------------
# 3️⃣ CREATE VECTOR STORE
# -------------------------------

def create_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embedding_model)


# -------------------------------
# 4️⃣ LOAD LIGHTWEIGHT LLM
# -------------------------------

def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        device=-1  # CPU
    )
    return HuggingFacePipeline(pipeline=pipe)


# -------------------------------
# 5️⃣ ASK QUESTION
# -------------------------------

def ask_question(vectorstore, llm, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.
Answer clearly using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)



# -------------------------------
# MAIN PROGRAM
# -------------------------------

if __name__ == "__main__":

    video_url = input("Enter YouTube URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]

    print("\nFetching transcript...")
    transcript = get_transcript(video_id)

    print("Splitting text...")
    docs = split_text(transcript)

    print("Creating vector database...")
    vectorstore = create_vectorstore(docs)

    print("Loading local LLM (TinyLlama)...")
    llm = load_llm()

    print("\nReady! Ask questions about the video.\n")

    while True:
        question = input("Your Question (or type exit): ")

        if question.lower() == "exit":
            break

        answer = ask_question(vectorstore, llm, question)
        print("\nAnswer:\n", answer)
        print("-" * 60)
