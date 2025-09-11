import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

# -------------------
# Load documents
loader = PyPDFLoader("hr_docs/sample_hr_policy_1.pdf")
docs = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)

# Embeddings model (Retriever)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector database
vectorstore = FAISS.from_documents(texts, embeddings)

# Generator model (Answering)
model_name = "google/flan-t5-small" 
hf_pipeline = pipeline("text2text-generation", model=model_name, max_length=300)

# Wrap pipeline in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# LangChain RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # retrieve top-2 docs
    return_source_documents=True
)

# -------------------
# Streamlit interface
st.title("HR Assistant Chatbot")
query = st.text_input("Ask a question about HR policies:")

if query:
    result = qa(query)
    st.write("**Answer:**", result["result"])

    # Show retrieved sources (for transparency)
    with st.expander("See retrieved documents"):
        for doc in result["source_documents"]:
            st.write(doc.page_content[:300] + "...")
