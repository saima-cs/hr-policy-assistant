# 📘 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about its contents.

The application retrieves the most relevant sections from the uploaded policy and uses Groq to generate an answer.

---

## 🚀 Features

- Upload HR Policy PDF
- Extract text from PDF
- Split policy into smaller chunks
- Generate embeddings using Sentence Transformers
- Semantic search using NumPy cosine similarity
- Generate answers using Groq
- Display retrieved policy sources
- Show policy page numbers
- Streamlit web interface
- Deployable on Streamlit Community Cloud

---

## 🧠 RAG Architecture

The application follows this workflow:

PDF Upload
↓
PDF Text Extraction
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
NumPy Similarity Search
↓
Relevant Policy Sections
↓
Groq LLM
↓
Final Answer

---

## 🛠 Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- NumPy
- Groq
- openai/gpt-oss-20b

---

## 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
