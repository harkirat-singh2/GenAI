import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# 1️⃣ Load Environment Variables
# -------------------------
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

# -------------------------
# 2️⃣ Initialize Pinecone
# -------------------------
pc = Pinecone(api_key=api_key)

index_name = "sample-index"

# -------------------------
# 3️⃣ Create Index if Not Exists
# -------------------------
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # MUST match embedding model dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(index_name)

# -------------------------
# 4️⃣ Create Embedding Model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 5️⃣ Create Vector Store
# -------------------------
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)

# -------------------------
# 6️⃣ Create Documents
# -------------------------
documents = [
    Document(page_content="LangChain is an LLM framework.", metadata={"id": 1}),
    Document(page_content="Pinecone is a cloud vector database.", metadata={"id": 2}),
    Document(page_content="Embeddings convert text into vectors.", metadata={"id": 3}),
]

# -------------------------
# 7️⃣ Add Documents
# -------------------------
vector_store.add_documents(documents)

print("✅ Documents added successfully!")

# -------------------------
# 8️⃣ Similarity Search
# -------------------------
results = vector_store.similarity_search("vector database", k=2)

print("\n🔎 Search Results:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 40)
