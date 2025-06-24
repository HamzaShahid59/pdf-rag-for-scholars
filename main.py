import streamlit as st
from services.rag_chain import create_rag_chain
from services.retriever import PineconeRetrieverWithThreshold
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up retriever and RAG chain
retriever = PineconeRetrieverWithThreshold()
rag_chain = create_rag_chain(retriever)

# Set app title
st.set_page_config(page_title="Smart Assistant | Atlas Copco", layout="wide")
st.title("🧠 Smart Assistant for Atlas Copco Documents")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
query = st.text_input("Ask something about Atlas Copco documents:", key="user_input")

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

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🕘 Chat History")
    for entry in st.session_state.chat_history:
        st.markdown(f"- {entry}")
