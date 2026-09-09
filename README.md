# 📚 HR Policy Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload an HR Policy PDF and ask questions about its contents.

The application retrieves relevant sections from the uploaded document using semantic search and generates answers using Groq.

## 🚀 Features

- Upload HR Policy PDF
- Extract PDF text with PyMuPDF
- Split documents into overlapping chunks
- Generate embeddings with Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers with Groq
- Uses `openai/gpt-oss-20b`
- Shows retrieved source sections
- Displays page numbers
- Streamlit web interface
- No external database required

## 🧠 RAG Architecture

```text
                HR Policy PDF
                      |
                      v
                PyMuPDF
                      |
                      v
              Extracted Text
                      |
                      v
                 Chunking
                      |
                      v
          Sentence Transformers
                      |
                      v
                Embeddings
                      |
                      v
                   FAISS
                Vector Index
                      |
                      |
User Question ------+
      |
      v
Question Embedding
      |
      v
FAISS Similarity Search
      |
      v
Relevant Policy Chunks
      |
      v
Groq / GPT-OSS-20B
      |
      v
Final HR Policy Answer
