"""
app.py
------
Streamlit frontend for the RAG Website Q&A pipeline.

Usage:
    streamlit run app.py
"""

import streamlit as st
from rag_pipeline import scrape_website, process_text, ask_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Website Q&A",
    page_icon="🔍",
    layout="centered",
)

st.title("RAG Website Q&A")
st.markdown(
    "Scrape any website, build a knowledge base, and ask questions — "
    "answers are grounded **only** in the scraped content."
)

# ── Session state init ─────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "source_url" not in st.session_state:
    st.session_state.source_url = None

# ── Sidebar: Website ingestion ─────────────────────────────────────────────────
st.sidebar.header("1 — Ingest Website")
url = st.sidebar.text_input("Website URL", placeholder="https://example.com")

if st.sidebar.button("Process Website", type="primary"):
    if not url:
        st.sidebar.error("Please enter a URL.")
    else:
        with st.sidebar.status("Processing...", expanded=True) as status:
            try:
                # Step 1: Scrape
                st.write("Scraping website...")
                text = scrape_website(url)
                st.write(f"Extracted **{len(text):,}** characters.")

                # Step 2: Chunk + Embed + Store
                st.write("Chunking & embedding text...")
                vector_store = process_text(text)

                # Save to session
                st.session_state.vector_store = vector_store
                st.session_state.source_url = url

                status.update(label="Website processed!", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.sidebar.error(f"Failed: {e}")

# Show active source
if st.session_state.source_url:
    st.sidebar.success(f"Active source: {st.session_state.source_url}")

# ── Main area: Question answering ─────────────────────────────────────────────
st.header("2 — Ask a Question")

query = st.text_input(
    "Your question",
    placeholder="What is this website about?",
    disabled=st.session_state.vector_store is None,
)

if st.button("Get Answer", disabled=st.session_state.vector_store is None):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving & generating answer..."):
            try:
                answer, chunks = ask_question(
                    query, st.session_state.vector_store
                )

                # Display answer
                st.subheader("Answer")
                st.markdown(answer)

                # Display retrieved chunks in an expandable section
                with st.expander("Retrieved Context Chunks", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk.page_content[:500])
                        st.divider()
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.vector_store is None:
    st.info("Process a website first using the sidebar to start asking questions.")
