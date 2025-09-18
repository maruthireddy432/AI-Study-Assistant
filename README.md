# 🤖 EduBot – AI Study Assistant

EduBot is an AI-powered personal study assistant built with Streamlit, LangChain, FAISS, HuggingFace embeddings, and Groq LLMs.
It allows you to upload study material (PDF or TXT), creates a knowledge base, and answers your questions only from the uploaded content using a Retrieval-Augmented Generation (RAG) pipeline.

# 🚀 Features

- 📂 Upload PDF or TXT files

- 🔍 Splits documents into chunks for efficient retrieval

- 🧠 Builds a FAISS vector store with HuggingFace embeddings

- 🤝 Integrates Groq LLMs for fast and accurate responses

- 💬 Chat interface with conversation history

- 🎯 Answers are strictly based on uploaded content (no hallucination)

# 🛠️ Tech Stack

- Frontend/UI: Streamlit
- LLM Inference: Groq (LLMs like gpt-oss-20b / llama3-70b)
- Framework: LangChain
- Vector Database: FAISS
- Embeddings: HuggingFace (all-MiniLM-L6-v2)
- Environment Management: Python + dotenv

# 🔑 Setup

- Add your Groq API Key to a .env file:

- GROQ_API_KEY=your_api_key_here

You can get a free API key at 👉 Groq Console

# ▶️ Run the App
- streamlit run app.py

# 📌 Usage

- Upload your study material (PDF or TXT).
- EduBot processes the file and builds a knowledge base.
- Ask questions in the chatbox.
- Get concise, factual answers only from your document.
