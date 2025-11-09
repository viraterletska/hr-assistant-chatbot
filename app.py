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
PERSIST_DIR = "chroma_db"

vectorstore_policies = Chroma(
    collection_name="hr_policies",
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

# Build RAG chain using LCEL
rag_chain_policies = (
    {"context": retriever_policies | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_code = (
    {"context": retriever_code | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------
# Chatbot logic
def chat_with_bot(user_message, chat_history):
    try:
        # Get answer from Labour Code
        answer = rag_chain_code.invoke(user_message)
        
        # Get source documents for display
        source_docs = retriever_code.invoke(user_message)
        
        if source_docs:
            source_text = "\n\n📄 Source (Labour Code):\n" + source_docs[0].page_content[:200] + "..."
        else:
            source_text = "\n\n(No specific source found)"
        
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