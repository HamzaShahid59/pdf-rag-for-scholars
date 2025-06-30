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
        You are a multilingual question-answering assistant.

        Use only the provided context to answer the query. Follow these rules strictly:

        1. If the context is relevant, use it to answer naturally.
        2. If the context is empty or irrelevant, say: "I don't know about this query".
        3. Do not use prior knowledge — only answer from the context.
        4. Answer in the **same language** as the user query (language code: {language}).
        5. Be clear, concise, and natural in that language.
        6. Do not add greetings or explanations unless they are part of the answer.

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
        translated_query = inputs.get("translated_query")
        original_language = inputs.get("language", "en")
        original_query = inputs.get("input")

        retrieved_docs = history_aware_retriever.invoke({
            "input": translated_query,
            "chat_history": inputs.get("chat_history", [])
        })

        answer = question_answer_chain.invoke({
            "input": original_query,              # original user query (in any language)
            "context": retrieved_docs,
            "chat_history": inputs.get("chat_history", []),
            "language": original_language         # pass language to the prompt
        })

        return {
            "answer": answer,
            "sources": retrieved_docs
        }


    return rag_chain_with_sources
