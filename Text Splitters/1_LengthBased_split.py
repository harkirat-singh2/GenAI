from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = '''Space exploration has led to incredible scientific discoveries. From landing on the Moon to
exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to
advancements in technology here on Earth. Satellite communications, GPS, and even certain medical
imaging techniques trace their roots back to innovations driven by space programs.
'''

loader = PyPDFLoader(r'C:\Users\Harkirat singh\Desktop\LANGCHAIN_MODELS\Langchain Document Loader\dl-curriculum.pdf')

docs=loader.load()

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 100,
    chunk_overlap = 0,
)

result = (splitter.split_documents(docs))

print(result[0].page_content)