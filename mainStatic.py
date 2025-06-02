from dotenv import load_dotenv
from services.rag_chain import create_rag_chain
from services.add_data import create_embeddings
from services.retriever import PineconeRetrieverWithThreshold
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_pdf():
    loader = PyPDFLoader("215_dil.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pages)
    return documents

def init_components():
    print("In Initialization")
    documents = load_pdf()
    create_embeddings(documents)
    retriever = PineconeRetrieverWithThreshold()
    return retriever

if __name__ == "__main__":
    retriever = init_components()
    rag_chain = create_rag_chain(retriever)

    query = "tell me about Natural frequencies comparison for point 5 reference"
    response = rag_chain.invoke({
        "input": query,
        "chat_history": []
    })

    print("\nAnswer:\n", response["answer"])
