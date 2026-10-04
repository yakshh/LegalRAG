"""Web demo.  Run:  streamlit run app.py"""
import time

import streamlit as st

from generate import answer
from retrieve import retrieve

NAMES = {"dense": "Dense", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + Reranking"}

st.title("LegalRAG : Legal Question Answering")
st.caption("This system is for educational and research purposes only and does not provide legal advice.")

question = st.text_input("Your question")
method = st.selectbox("Method", list(NAMES), format_func=NAMES.get)

if st.button("Ask") and question.strip():
    start = time.time()
    with st.spinner("Searching the Acts..."):
        chunks = retrieve(question, method)
    retrieval_time = time.time() - start

    st.subheader("Answer")
    try:
        with st.spinner("Writing the answer..."):
            st.markdown(answer(question, chunks))
    except Exception as e:  # e.g. no internet, bad key, Gemini busy
        st.error(f"Could not get an answer from Gemini ({e}). The retrieved evidence is shown below.")
    st.caption(f"Retrieval time: {retrieval_time:.2f} s | Total response time: {time.time() - start:.2f} s")

    st.subheader("Evidence")
    for c in chunks:
        with st.expander(f"{c['document']} | page {c['page']} | {c['section']}"):
            st.text(c["text"])
