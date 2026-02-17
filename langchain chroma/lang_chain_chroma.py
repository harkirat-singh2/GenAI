from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# 1️⃣ Create Documents
# -------------------------
doc1 = Document(
    page_content="LangChain is a framework for building LLM-powered applications.",
    metadata={"id": 1}
)

doc2 = Document(
    page_content="Chroma is a vector database used to store embeddings.",
    metadata={"id": 2}
)

doc3 = Document(
    page_content="Embeddings convert text into numerical vectors.",
    metadata={"id": 3}
)

doc4 = Document(
    page_content="Retrieval Augmented Generation improves LLM accuracy.",
    metadata={"id": 4}
)

doc5 = Document(
    page_content="HuggingFace provides open-source embedding models.",
    metadata={"id": 5}
)

documents = [doc1, doc2, doc3, doc4, doc5]

# -------------------------
# 2️⃣ Create Embedding Model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 3️⃣ Create / Load Chroma DB
# -------------------------
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db",
    collection_name="sample"
)

# -------------------------
# 4️⃣ Add Documents
# -------------------------
# vector_store.add_documents(documents)

# # -------------------------
# # 5️⃣ Check Stored Data
# # -------------------------
# data = vector_store.get(include=["documents", "metadatas"])
# print("Total documents stored:", vector_store._collection.count())
# print(data)

results = vector_store.similarity_search(
    query="Best Embedding model",
    k=3
)

result2 = vector_store.similarity_search_with_score(
    query="Best Embedding model",
    k=3
)

# meta-data filtering
vector_store.similarity_search_with_score(
query="",
filter={"team": "Chennai Super Kings"}

)

# update documents
updated_doc1 = Document(
page_content="Best player",
metadata={"team": "Royal Challengers Bangalore"}

)

vector_store. update_document(document_id='1', document=updated_doc1)


# delete document
vector_store.delete(ids=['1'])

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas' ])


for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
