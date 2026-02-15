from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
  path='Books',
  glob='**/*.txt',
  loader_cls=TextLoader
)

docs = loader.load()

print(docs)