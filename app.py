import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Chroma collections with embedding function
PERSIST_DIR = "chroma_db_hr"
vectorstore_policies = Chroma(
    collection_name="hr_faq_and_guidelines",
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
vectorstore_code = Chroma(
    collection_name="labour_code",
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# -------------------
# LLM (Ollama running locally)
llm = OllamaLLM(model="llama3")

# -------------------
# Modern RAG Chain using LCEL
def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Create prompt template
template = """You are an HR assistant. Answer the question based on the following context from company policies and labour code.

Context:
{context}

Question: {question}

Answer in a clear and professional manner. If you're not sure, say so."""

prompt = ChatPromptTemplate.from_template(template)

# Create retrievers
retriever_policies = vectorstore_policies.as_retriever(search_kwargs={"k": 2})
retriever_code = vectorstore_code.as_retriever(search_kwargs={"k": 2})

# -------------------
# Store retrieved documents globally for display
retrieved_docs = {"policies": [], "code": []}

def retrieve_from_both_sources(question):
    """Retrieve documents from HR FAQ first, then Labour Code"""
    global retrieved_docs
    
    # Get documents from HR FAQ
    docs_policies = retriever_policies.invoke(question)
    
    # Get documents from Labour Code
    docs_code = retriever_code.invoke(question)
    
    # Store for later display
    retrieved_docs["policies"] = docs_policies
    retrieved_docs["code"] = docs_code
    
    # Combine documents (HR FAQ first, then Labour Code)
    all_docs = docs_policies + docs_code
    
    return all_docs

def format_sources_from_retrieved():
    """Format source information from actually retrieved documents"""
    sources = []
    
    if retrieved_docs["policies"]:
        # Only show if there's meaningful content
        content = retrieved_docs["policies"][0].page_content[:200].strip()
        if content:
            sources.append(f"📋 HR FAQ & Guidelines:\n{content}...")
    
    if retrieved_docs["code"]:
        content = retrieved_docs["code"][0].page_content[:200].strip()
        if content:
            sources.append(f"📄 Labour Code:\n{content}...")
    
    if sources:
        return "\n\n" + "\n\n".join(sources)
    else:
        return "\n\n(No specific source found)"

# Build RAG chain with combined retrieval
def get_context(question):
    """Custom function to retrieve and format context from both sources"""
    all_docs = retrieve_from_both_sources(question)
    return format_docs(all_docs)

rag_chain = (
    {"context": get_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------
# Chatbot logic
def chat_with_bot(user_message, chat_history):
    try:
        # Get answer from both sources (this also populates retrieved_docs)
        answer = rag_chain.invoke(user_message)
        
        # Format sources from the documents that were actually used
        source_text = format_sources_from_retrieved()
        bot_message = answer + source_text
        
    except Exception as e:
        bot_message = f"Sorry, I encountered an error: {str(e)}"
    
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_message})
    
    return chat_history

# -------------------
# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💼 HR Assistant Chatbot")
    gr.Markdown("Ask questions about HR policies and Lithuanian Labour Code")
    
    chatbot = gr.Chatbot(type="messages", height=500)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about HR policies or Labour Code...",
            show_label=False,
            lines=1,
            max_lines=5,
            scale=9
        )
        send_btn = gr.Button("➤", scale=1, variant="primary")
    
    def user_submit(user_message, chat_history):
        if not user_message.strip():
            return "", chat_history
        chat_history = chat_with_bot(user_message, chat_history)
        return "", chat_history
    
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot])

# -------------------
# Run app
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)