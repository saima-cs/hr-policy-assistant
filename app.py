import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📘 HR Policy Assistant")
st.write(
    "Upload an HR Policy PDF and ask questions about its contents."
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


groq_client = get_groq_client()


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def create_chunks(pages, chunk_size=900, overlap=150):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

def create_embeddings(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# ---------------------------------------------------------
# RETRIEVAL USING NUMPY
# ---------------------------------------------------------

def retrieve_chunks(query, chunks, embeddings, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Cosine similarity because embeddings are normalized
    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index])
        })

    return results


# ---------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------

def generate_answer(question, retrieved_chunks):

    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            f"[Page {result['page']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question ONLY using the HR policy context provided below.

If the answer is not contained in the context, clearly say:

"I could not find this information in the uploaded HR policy."

Do not invent policies or information.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and concise answer.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided HR policy context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# API KEY CHECK
# ---------------------------------------------------------

if groq_client is None:

    st.error(
        "⚠️ GROQ_API_KEY is missing. "
        "Add it in Streamlit Cloud → Manage app → Settings → Secrets."
    )

    st.stop()


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"]
)


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file:

    with st.spinner("Reading HR policy..."):

        pages = extract_pdf_text(uploaded_file)

    if not pages:

        st.error(
            "No readable text was found in this PDF."
        )

        st.stop()

    with st.spinner("Creating document chunks..."):

        chunks = create_chunks(pages)

    with st.spinner("Creating embeddings..."):

        embeddings = create_embeddings(chunks)

    st.success(
        f"✅ PDF processed successfully — {len(chunks)} chunks created."
    )

    st.divider()

    # -----------------------------------------------------
    # QUESTION
    # -----------------------------------------------------

    question = st.text_input(
        "💬 Ask a question about the HR policy"
    )

    if question:

        with st.spinner("Searching the policy..."):

            retrieved_chunks = retrieve_chunks(
                question,
                chunks,
                embeddings,
                top_k=5
            )

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                retrieved_chunks
            )

        st.subheader("🤖 Answer")

        st.write(answer)

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.subheader("📚 Retrieved Sources")

        for i, result in enumerate(retrieved_chunks, start=1):

            with st.expander(
                f"Source {i} — Page {result['page']}"
            ):

                st.write(result["text"])

                st.caption(
                    f"Similarity score: {result['score']:.3f}"
                )

else:

    st.info(
        "👆 Upload an HR Policy PDF to get started."
    )
