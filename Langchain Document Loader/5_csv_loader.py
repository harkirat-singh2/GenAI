from langchain_community.document_loaders import TextLoader,CSVLoader

loader = CSVLoader('data.csv')

docs = loader.load()

print(docs)

