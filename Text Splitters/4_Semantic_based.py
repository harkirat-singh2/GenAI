from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Create embeddings FIRST
embeddings = HuggingFaceEmbeddings()

# Step 2: Create semantic splitter
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80  # Try 80 or 95
)

text = """Farmers were working hard...
The IPL is the biggest cricket league...
Terrorism is a big danger...
"""

# Step 3: Split text
chunks = text_splitter.split_text(text)

print(len(chunks))
for chunk in chunks:
    print("\n", chunk)
