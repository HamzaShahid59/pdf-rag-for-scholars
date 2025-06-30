from typing import List
import streamlit as st
from services.rag_chain import create_rag_chain
from services.retriever import PineconeRetrieverWithThreshold, pc, INDEX_NAME
from dotenv import load_dotenv
from services.translate import detect_and_translate

# Load environment variables
load_dotenv()

# Config
st.set_page_config(page_title="Smart Assistant | Atlas Copco", layout="wide")
st.markdown("## 🧠 Smart Assistant for Atlas Copco Documents")

# ---------------------------
# Fetch and cache namespaces
# ---------------------------
if "namespace_list" not in st.session_state:
    index = pc.Index(INDEX_NAME)
    fetched_namespaces = list(index.describe_index_stats().namespaces.keys())
    st.session_state.namespace_list = [ns for ns in fetched_namespaces if ns.lower() != "__default__"]

namespaces = st.session_state.namespace_list

# ---------------------------
# State setup
# ---------------------------
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "active_modal" not in st.session_state:
    st.session_state.active_modal = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Utility - get files for namespace
# ---------------------------
def get_unique_files_in_namespace(namespace: str, top_k: int = 100) -> List[str]:
    dummy_vector = [0.0] * 1536
    index = pc.Index(INDEX_NAME)
    results = index.query(
        vector=dummy_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    filenames = set()
    for match in results.matches:
        filename = match.metadata.get("filename") or match.metadata.get("source")
        if filename:
            filenames.add(filename)
    return sorted(filenames)

# ---------------------------
# UI Tabs
# ---------------------------
tab1, tab2 = st.tabs(["💬 Ask Question", "📁 View PDFs"])

# ---------------------------
# Tab 1: Ask Question
# ---------------------------
with tab1:
    st.subheader("Select a Category to Ask Questions")

    cols = st.columns(len(namespaces))
    for i, ns in enumerate(namespaces):
        with cols[i]:
            is_selected = (st.session_state.selected_namespace == ns)
            border_color = "#4da6ff" if is_selected else "#ccc"
            if st.button(f"📦 {ns}", key=f"ask_card_{ns}"):
                st.session_state.selected_namespace = ns

            st.markdown(f"""
                <style>
                div[data-testid="column"] > div:has(button[kind='secondary'][key='ask_card_{ns}']) {{
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                }}
                </style>
            """, unsafe_allow_html=True)

    selected_ns = st.session_state.selected_namespace

    if selected_ns:
        retriever = PineconeRetrieverWithThreshold(namespace=selected_ns)
        rag_chain = create_rag_chain(retriever)

        query = st.text_input(f"Ask something in `{selected_ns}`:", key="user_input")

        if query:
            with st.spinner("Thinking..."):
                translated_input = detect_and_translate(query)
                response = rag_chain({
                    "input": query,
                    "translated_query": translated_input["translated_query"],
                    "language": translated_input["language"],
                    "chat_history": st.session_state.chat_history
                })

            st.session_state.chat_history.append(f"User: {query}")
            st.session_state.chat_history.append(f"Bot: {response['answer']}")

            st.subheader("Answer:")
            st.write(response["answer"])

            if response.get("sources"):
                refs = set()
                for doc in response["sources"]:
                    filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    refs.add(f"{filename} (page {page})")

                st.markdown("#### 📚 Relevant Sources Checked:")
                for ref in sorted(refs):
                    st.markdown(f"- {ref}")

        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("🕘 Chat History")
            for entry in st.session_state.chat_history:
                st.markdown(f"- {entry}")

    else:
        st.info("Please select a namespace to begin.")

# ---------------------------
# Tab 2: View PDFs
# ---------------------------
with tab2:
    st.subheader("View PDFs in a Category")

    cols = st.columns(len(namespaces))
    for i, ns in enumerate(namespaces):
        with cols[i]:
            is_selected = (st.session_state.active_modal == ns)
            border_color = "#4da6ff" if is_selected else "#ccc"
            if st.button(f"📦 {ns}", key=f"view_card_{ns}"):
                st.session_state.active_modal = ns

            st.markdown(f"""
                <style>
                div[data-testid="column"] > div:has(button[kind='secondary'][key='view_card_{ns}']) {{
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                }}
                </style>
            """, unsafe_allow_html=True)

    active_ns = st.session_state.active_modal

    if active_ns:
        files = get_unique_files_in_namespace(active_ns)
        with st.expander(f"📁 Files in `{active_ns}`", expanded=True):
            if files:
                for file in files:
                    st.markdown(f"- {file}")
            else:
                st.info("No PDFs found in this namespace.")
            st.button("Close", key=f"close_{active_ns}", on_click=lambda: st.session_state.update({"active_modal": None}))
