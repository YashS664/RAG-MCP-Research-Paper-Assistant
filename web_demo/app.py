import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import rag_utils

st.set_page_config(page_title="Research Paper RAG Search", page_icon="📚", layout="centered")

st.title("📚 Research Paper Q&A")
st.caption(
    "Ask a question about 5 papers on RAG agents and tool use -- "
    "answers are generated only from the papers, with citations.."
)

# --- Sidebar: show which papers are indexed ---
with st.sidebar:
    st.header("Indexed papers")
    try:
        rag_utils._load_resources()
        papers = sorted(set(m["paper"] for m in rag_utils._metadata))
        for p in papers:
            st.write(f"- {p}")
    except FileNotFoundError:
        st.error("No index found. Run `python ingest.py` first.")

# --- Main search box ---
query = st.text_input(
    "Ask a question", placeholder="e.g. How does Self-RAG differ from the original RAG paper?")

col1, col2 = st.columns([1, 1])
with col1:
    ask_clicked = st.button("Get answer", type="primary", use_container_width=True)
with col2:
    search_clicked = st.button("Raw search (no LLM)", use_container_width=True)

if ask_clicked and query.strip():
    with st.spinner("Retrieving relevant excerpts and generating an answer..."):
        try:
            result = rag_utils.answer_question(query)
            st.markdown('### Answer')
            st.write(result["answer"])

            st.markdown("###n Sources")
            for s in result["sources"]:
                st.write(f"- **{s['paper']}** (similarity: {s['score']})")
        except FileNotFoundError:
            st.error("No index found. Run python ingest.py first")
        except Exception as e:
            st.error(f"Something went wrong {e}")

elif search_clicked and query.strip():
    with st.spinner("Searching.."):
        try:
            results = rag_utils.retrieve(query, top_k=5)
            if not results:
                st.warning("No relevant excerpts found.")
            else:
                st.markdown("### Matching excerpts")
                for r in results:
                    with st.expander(
                        f"[{r['paper']} - chunk {r['chunk_index']} (score{r['score']:.3f})"
                    ):
                        st.write(r["text"])

        except FileNotFoundError:
            st.error("No index found. Run `python ingest.py` first.")

elif (ask_clicked or search_clicked) and not query.strip():
    st.warning("Type a question first.")
