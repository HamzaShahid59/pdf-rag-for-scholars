import streamlit as st
from services.rag_chain import create_rag_chain
from services.add_data import create_embeddings
from services.retriever import PineconeRetrieverWithThreshold
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Set up retriever and RAG chain
retriever = PineconeRetrieverWithThreshold()
rag_chain = create_rag_chain(retriever)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Page", ["Ask a Question" ,"Upload PDF" ])


# ------------------ Ask Question Page ------------------
if app_mode == "Ask a Question":
    st.title("💬 Ask Questions")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Enter your question", key="user_input")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain({
                "input": query,
                "chat_history": st.session_state.chat_history
            })

        st.session_state.chat_history.append(f"User: {query}")
        st.session_state.chat_history.append(f"Bot: {response['answer']}")

        st.subheader("Answer:")
        st.write(response["answer"])

        # Show references
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
        st.subheader("Chat History")
        for entry in st.session_state.chat_history:
            st.markdown(f"- {entry}")


# ------------------ Upload PDF Page ------------------
elif app_mode == "Upload PDF":
    st.title("📄 Upload PDF for Embedding")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if not uploaded_file.name.endswith(".pdf"):
            st.error("Only PDF files are allowed.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.info("Processing PDF...")

            try:
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                documents = text_splitter.split_documents(pages)

                create_embeddings(documents)

                st.success(f"{uploaded_file.name} processed and embedded successfully.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                os.remove(tmp_file_path)

