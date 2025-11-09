# 🤖 HR Assistant Chatbot

> An intelligent AI-powered chatbot that answers HR policy and Lithuanian Labour Code questions using RAG technology.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

- 🤖 **AI-Powered**: Uses Llama 3 via Ollama for natural language understanding
- 🔍 **Smart Search**: RAG architecture with semantic search via ChromaDB
- 🇱🇹 **Lithuanian Compliant**: Built-in Labour Code knowledge
- 💬 **Interactive UI**: Clean Gradio interface with chat history
- 📚 **Source Citations**: Transparent responses with document references
- 🔒 **Privacy First**: Runs completely locally on your machine

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download Llama 3
ollama pull llama3
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hr-assistant-chatbot.git
cd hr-assistant-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# 1. Embed documents (first time only)
python embed_docs.py

# 2. Launch chatbot
python app.py
```

🎉 **Done!** Open http://localhost:7860 in your browser.

---

## 📁 Project Structure

```
hr-assistant-chatbot/
├── app.py              # Main chatbot application
├── embed_docs.py       # Document embedding script
├── requirements.txt    # Python dependencies
├── hr_docs/           # PDF documents folder
│   ├── sample_hr_policy_2.pdf
│   └── LithuaniaLabourCode.pdf
└── chroma_db_new/     # Vector database (auto-generated)
```

---

## 💡 Usage Examples

**Question:** What is the probation period in Lithuania?

**Answer:** According to the Lithuanian Labour Code, the probation period cannot exceed 3 months. During this period, either party may terminate the employment contract with 2 weeks' notice...

📄 *Source: Labour Code, Article 35*

---

## ⚙️ Configuration

### Change LLM Model

```python
# app.py
llm = OllamaLLM(model="mistral")  # Options: llama2, mistral, codellama
```

### Adjust Retrieval

```python
# Retrieve more documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # default: 2
```

### Modify Chunking

```python
# embed_docs.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # default: 800
    chunk_overlap=100   # default: 50
)
```

---

## 🛠️ Troubleshooting

**Ollama not running?**
```bash
ollama serve
ollama list
```

**Import errors?**
```bash
pip install --upgrade -r requirements.txt
```

**Slow responses?**
- Use smaller model: `ollama pull llama2:7b`
- Reduce retrieval: `search_kwargs={"k": 1}`

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (Llama 3) |
| Framework | LangChain LCEL |
| Embeddings | Sentence Transformers |
| Vector DB | ChromaDB |
| UI | Gradio |

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This chatbot is for informational purposes only. Always verify critical information with qualified HR professionals or legal experts.

---