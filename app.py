import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------
# Load and split PDF
loader = PyPDFLoader("hr_docs/sample_hr_policy_2.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
texts = splitter.split_documents(docs)

# Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

# Generator model
model_name = "google/flan-t5-small"
hf_pipeline = pipeline("text2text-generation", model=model_name, max_length=300)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# -------------------
# Gradio streaming function
def stream_response(message, history):
    if message is None:
        return

    # Retrieve relevant documents
    docs = vectorstore.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Construct RAG prompt
    rag_prompt = f"""
    You are an assistant that answers questions using only the provided knowledge.
    Do not use your internal knowledge.
    Question: {message}
    Conversation history: {history}
    Knowledge: {knowledge}
    """

    # Stream response from LLM
    partial_message = ""
    for token in llm.stream(rag_prompt):
        partial_message += token
        yield partial_message

# -------------------
# Gradio ChatInterface
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask a question about HR policies...",
        container=False,
        autoscroll=True,
        scale=7
    )
)

# Launch app
chatbot.launch()
