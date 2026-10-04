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

if st.button("Ask") and question:
    start = time.time()
    chunks = retrieve(question, method)
    st.subheader("Answer")
    st.write(answer(question, chunks))
    st.caption(f"Response time: {time.time() - start:.2f} s")
    st.subheader("Evidence")
    for c in chunks:
        with st.expander(f"{c['document']} | page {c['page']} | {c['section']}"):
            st.write(c["text"])
