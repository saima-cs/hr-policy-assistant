import streamlit as st
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("📘 HR Policy Assistant")
st.markdown(
    "Upload an HR Policy PDF and ask questions using AI-powered RAG."
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading AI model..."):
    model = load_model()


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return Groq(api_key=api_key)

    except Exception:
        return None


client = get_groq_client()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    pdf_data = uploaded_file.getvalue()

    document = fitz.open(
        stream=pdf_data,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text and text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages


# =========================================================
# CREATE TEXT CHUNKS
# =========================================================

def create_chunks(pages):

    chunks = []

    chunk_size = 800
    overlap = 100

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


# =========================================================
# SEARCH RELEVANT CHUNKS
# =========================================================

def search_policy(question, chunks, embeddings):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    similarities = np.dot(
        embeddings,
        question_embedding
    )

    top_k = min(5, len(chunks))

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": similarities[index]
        })

    return results


# =========================================================
# ASK GROQ
# =========================================================

def ask_groq(question, results):

    context = ""

    for result in results:

        context += (
            f"\n\n--- Page {result['page']} ---\n"
            f"{result['text']}"
        )

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context below.

Do not make up information.

If the answer is not available in the uploaded policy, say:

"I could not find this information in the uploaded HR policy."

Keep the answer clear and easy to understand.

HR POLICY:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful HR policy assistant. "
                    "Use only the provided policy context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=600
    )

    return response.choices[0].message.content


# =========================================================
# GROQ KEY CHECK
# =========================================================

if client is None:

    st.error(
        "⚠️ GROQ API key is missing."
    )

    st.info(
        "Go to Streamlit Cloud → Manage app → Settings → Secrets "
        "and add your GROQ_API_KEY."
    )

    st.stop()


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    st.success(
        f"📄 Uploaded: {uploaded_file.name}"
    )

    with st.spinner("Reading PDF..."):

        pages = extract_text_from_pdf(
            uploaded_file
        )

    if not pages:

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.stop()

    st.success(
        f"✅ Extracted text from {len(pages)} pages."
    )

    with st.spinner("Creating document chunks..."):

        chunks = create_chunks(pages)

    st.success(
        f"✅ Created {len(chunks)} text chunks."
    )

    with st.spinner("Creating embeddings..."):

        embeddings = create_embeddings(
            chunks
        )

    st.success(
        "✅ HR Policy is ready for questions!"
    )

    st.divider()

    # =====================================================
    # QUESTION BOX
    # =====================================================

    question = st.text_input(
        "💬 Ask a question about the HR policy",
        placeholder="Example: How many annual leaves are allowed?"
    )

    if question:

        with st.spinner(
            "🔎 Searching the HR policy..."
        ):

            results = search_policy(
                question,
                chunks,
                embeddings
            )

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = ask_groq(
                question,
                results
            )

        st.subheader("🤖 Answer")

        st.write(answer)

        st.divider()

        # =================================================
        # SOURCES
        # =================================================

        st.subheader("📚 Sources")

        for number, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Source {number} — Page {result['page']}"
            ):

                st.write(
                    result["text"]
                )

                st.caption(
                    f"Similarity: {result['score']:.3f}"
                )

else:

    st.info(
        "👆 Upload an HR Policy PDF to begin."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HR Policy Assistant • RAG • Streamlit • "
    "Sentence Transformers • Groq"
)
