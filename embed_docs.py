import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------
# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

# -------------------
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -------------------
# ChromaDB directory
persist_dir = "chroma_db"
os.makedirs(persist_dir, exist_ok=True)

# -------------------
# Function to embed a PDF into ChromaDB collection
def embed_pdf(pdf_path, collection_name):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    texts = splitter.split_documents(docs)
    
    collection = Chroma.from_documents(
        texts,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    collection.persist()
    print(f"Embedded and saved: {collection_name}")

# -------------------
# Embed HR policies
embed_pdf("hr_docs/sample_hr_policy_2.pdf", "hr_policies")

# Embed Lithuania Labour Code
embed_pdf("hr_docs/LithuaniaLabourCode.pdf", "labour_code")
