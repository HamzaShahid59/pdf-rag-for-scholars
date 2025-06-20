from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

def create_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o")

    # Step 1: Use history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", """Given the chat history above, formulate a standalone question that 
        would help answer the current question using the desired document context.""")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # Step 2: Contextual answering prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict question-answering assistant.

        Your job is to answer a user query using only the provided context. Follow these strict rules:

        1. If the context is empty, or does not contain relevant information about the query, say exactly: "I don't know about this query".
        2. Do not use prior knowledge to answer — only use what's in the context.
        3. Do not explain or paraphrase the context unless it is directly relevant to the query.
        4. Remove unnecessary line breaks or symbols. Output clean, human-readable text.
        5. Do not include any greetings, salutations, or extra commentary.

        Query:
        {input}

        Context:
        {context}
        """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )

    # Step 3: Custom wrapper chain
    def rag_chain_with_sources(inputs):
        retrieved_docs = history_aware_retriever.invoke(inputs)
        answer = question_answer_chain.invoke({
            "input": inputs["input"],
            "context": retrieved_docs,
            "chat_history": inputs.get("chat_history", [])
        })
        return {
            "answer": answer,
            "sources": retrieved_docs
        }

    return rag_chain_with_sources
