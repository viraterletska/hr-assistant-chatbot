import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------
# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

# -------------------
# ChromaDB directory
persist_dir = "chroma_db_hr"
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
    # Note: persist() is deprecated in newer versions, data is auto-saved
    print(f"Embedded and saved: {collection_name}")

# -------------------
# Embed HR policies
if __name__ == "__main__":
    embed_pdf("hr_docs/hr_faq_doc.pdf", "hr_faq_and_guidelines")
    # Embed Lithuania Labour Code
    embed_pdf("hr_docs/LithuaniaLabourCode.pdf", "labour_code")