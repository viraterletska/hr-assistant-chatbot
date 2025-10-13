import gradio as gr
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# Create RetrievalQA chains
qa_policies = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore_policies.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

qa_code = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore_code.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# -------------------
# Chatbot logic
def chat_with_bot(user_message, chat_history):
    # Try enterprise HR policies first
    # result = qa_policies.invoke({"query": user_message})

    # if result["source_documents"]:
    #     answer = result["result"]
    #     source_text = "\n\nSource (Company Policy):\n" + result["source_documents"][0].page_content[:200] + "..."
    # else:
    # Fallback to Labour Code
    result = qa_code.invoke({"query": user_message})
    answer = result["result"]
    source_text = "\n\nSource (Labour Code):\n" + result["source_documents"][0].page_content[:200] + "..."

    bot_message = answer + source_text

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return chat_history

# -------------------
# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'>HR Assistant Chatbot</h1>")

    chatbot = gr.Chatbot(type="messages")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about HR policies or Labour Code...",
            show_label=False,
            lines=1,
            max_lines=5,
            container=True
        )
        send_btn = gr.Button("➤", elem_id="send-button")

    def user_submit(user_message, chat_history):
        chat_history = chat_with_bot(user_message, chat_history)
        return "", chat_history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot])

# -------------------
# Run app
demo.launch(inbrowser=True, share=False)
