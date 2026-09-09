import os
import tempfile
from typing import List, Tuple

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f5f7fa;
            border: 1px solid #e1e5ea;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 8px;
            background-color: #fafafa;
            border-left: 4px solid #555;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets."
        )

    return Groq(api_key=api_key)


# ---------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of tuples: (page_number, page_text)
    """

    pages = []

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        document = fitz.open(temp_path)

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append((page_number, text))

        document.close()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return pages


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def create_chunks(
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Split extracted PDF text into overlapping chunks.

    Each chunk keeps its original page number.
    """

    chunks = []

    for page_number, text in pages:

        # Normalize whitespace
        text = " ".join(text.split())

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ---------------------------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------------------------

def build_faiss_index(chunks, model):
    """
    Generate embeddings and build a FAISS cosine-similarity index.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve_chunks(
    question: str,
    index,
    chunks,
    model,
    top_k: int = TOP_K,
):
    """
    Retrieve the most relevant chunks using FAISS.
    """

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


# ---------------------------------------------------------
# GENERATE ANSWER WITH GROQ
# ---------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks,
    client,
):
    """
    Generate an answer using only retrieved HR policy context.
    """

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the uploaded "
            "HR policy document."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY using the HR policy context
provided by the user.

Rules:

1. Use only the provided policy context.
2. Do not invent HR rules, benefits, leave policies, salaries,
   disciplinary procedures, or other information.
3. If the answer is not contained in the context, clearly say:
   "The uploaded HR policy does not contain enough information
   to answer this question."
4. Be concise and professional.
5. When possible, mention the relevant policy page number.
6. Do not make legal claims that are not explicitly stated in
   the uploaded document.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Answer the question using only the HR policy context above.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=2,
        max_value=8,
        value=TOP_K,
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        **1. Upload**
        
        Upload an HR Policy PDF.

        **2. Extract**
        
        PyMuPDF extracts the document text.

        **3. Chunk**
        
        The text is divided into smaller sections.

        **4. Embed**
        
        Sentence Transformers converts chunks into vectors.

        **5. Search**
        
        FAISS retrieves the most relevant sections.

        **6. Generate**
        
        Groq generates an answer using the retrieved policy text.
        """
    )

    st.divider()

    if st.session_state.document_name:

        st.success(
            f"Loaded: {st.session_state.document_name}"
        )

    else:

        st.info("No HR policy uploaded yet.")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📚 HR Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload an HR policy document and ask questions using
    Retrieval-Augmented Generation (RAG).
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# API CHECK
# ---------------------------------------------------------

try:
    embedding_model = load_embedding_model()
    groq_client = load_groq_client()

except Exception as error:

    st.error(str(error))

    st.stop()


# ---------------------------------------------------------
# PDF UPLOADER
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF.",
)


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner("Processing HR policy..."):

            pdf_bytes = uploaded_file.getvalue()

            pages = extract_pdf_text(pdf_bytes)

            if not pages:

                st.error(
                    "No readable text was found in this PDF. "
                    "If the PDF is scanned, OCR may be required."
                )

                st.stop()

            chunks = create_chunks(pages)

            if not chunks:

                st.error(
                    "Could not create text chunks from the PDF."
                )

                st.stop()

            index = build_faiss_index(
                chunks,
                embedding_model,
            )

            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []

        st.success(
            f"✅ Policy processed successfully: "
            f"{len(pages)} pages and {len(chunks)} chunks."
        )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

if st.session_state.faiss_index is not None:

    question = st.chat_input(
        "Ask a question about the HR policy..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching HR policy..."):

                retrieved = retrieve_chunks(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    embedding_model,
                    top_k=top_k,
                )

                answer = generate_answer(
                    question,
                    retrieved,
                    groq_client,
                )

            st.markdown(
                '<div class="answer-box">'
                + answer
                + "</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            st.subheader("📌 Retrieved Policy Sections")

            for i, item in enumerate(retrieved, start=1):

                with st.expander(
                    f"Source {i} — Page {item['page']} "
                    f"— Similarity {item['score']:.3f}"
                ):

                    st.write(item["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

else:

    st.info(
        "👆 Upload an HR Policy PDF above to start asking questions."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "HR Policy Assistant • RAG • FAISS • "
    "Sentence Transformers • Groq • Streamlit"
)
