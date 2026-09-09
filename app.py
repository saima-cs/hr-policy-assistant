import streamlit as st
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📘 HR Policy Assistant")
st.write(
    "Upload an HR Policy PDF and ask questions about the policy."
)

st.divider()


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return Groq(api_key=api_key)

    except Exception:
        return None


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number in range(len(document)):

        page = document[page_number]

        text = page.get_text("text").strip()

        if text:

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(pages):

    chunks = []

    chunk_size = 800
    overlap = 100

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )

            start += chunk_size - overlap

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):

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


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    chunks,
    embeddings,
    model,
    top_k=5
):

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

    number_of_results = min(
        top_k,
        len(chunks)
    )

    best_indices = np.argsort(
        similarities
    )[-number_of_results:][::-1]

    results = []

    for index in best_indices:

        results.append(
            {
                "text": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": float(similarities[index])
            }
        )

    return results


# ============================================================
# GROQ ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    client
):

    context = ""

    for result in results:

        context += (
            f"\n\nPAGE {result['page']}:\n"
            f"{result['text']}"
        )

    prompt = f"""
You are an HR Policy Assistant.

Answer the question using ONLY the HR policy context below.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the policy does not contain the answer, say:
   "I could not find this information in the uploaded HR policy."
4. Give a clear and concise answer.
5. Mention the relevant policy page when useful.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions strictly from the provided HR policy."
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


# ============================================================
# INITIALIZE
# ============================================================

try:

    embedding_model = load_embedding_model()

except Exception as error:

    st.error("❌ Could not load the embedding model.")

    st.exception(error)

    st.stop()


groq_client = load_groq_client()


if groq_client is None:

    st.error("❌ Groq API key is missing.")

    st.info(
        "Go to Streamlit Cloud → Manage app → Settings → Secrets "
        "and add GROQ_API_KEY."
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your HR Policy PDF",
    type=["pdf"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload an HR Policy PDF to start."
    )

else:

    st.success(
        f"📄 Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # EXTRACT PDF
    # --------------------------------------------------------

    with st.spinner("Reading your HR Policy PDF..."):

        pages = extract_pdf(uploaded_file)

    if not pages:

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.warning(
            "If your PDF is scanned images, OCR may be required."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    with st.spinner("Preparing policy information..."):

        chunks = create_chunks(pages)

    if not chunks:

        st.error(
            "❌ Could not create text chunks from the PDF."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    with st.spinner("Creating document embeddings..."):

        embeddings = create_embeddings(
            chunks,
            embedding_model
        )

    st.success(
        f"✅ Policy ready! {len(chunks)} sections indexed."
    )

    st.divider()

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_input(
        "💬 Ask a question",
        placeholder="Example: How many annual leaves can an employee take?"
    )

    if question:

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        with st.spinner("🔎 Searching the HR policy..."):

            results = retrieve_documents(
                question,
                chunks,
                embeddings,
                embedding_model,
                top_k=5
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.spinner("🤖 Generating answer..."):

            try:

                answer = generate_answer(
                    question,
                    results,
                    groq_client
                )

                st.subheader("🤖 Answer")

                st.write(answer)

            except Exception as error:

                st.error(
                    "❌ Could not generate the answer."
                )

                st.exception(error)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.divider()

        st.subheader("📚 Retrieved Sources")

        for number, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Source {number} — Page {result['page']}"
            ):

                st.write(result["text"])

                st.caption(
                    f"Similarity score: {result['score']:.3f}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Policy Assistant | RAG | Streamlit | "
    "Sentence Transformers | Groq"
)
