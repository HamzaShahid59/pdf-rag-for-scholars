from dotenv import load_dotenv
load_dotenv()

from typing import List, Literal, Optional, Dict, Any
import json
import streamlit as st

# LangChain / LLM
try:
    from langchain_openai import ChatOpenAI
except Exception:
    # Fallback for older LangChain installs
    from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

# Your services
from services.translate import detect_and_translate
from services.rag_chain import create_rag_chain
from services.retriever import PineconeRetrieverWithThreshold, pc, INDEX_NAME
from tools.fault_code_tool import FaultCodeTool
from tools.parts_tool import PartsTool


# ---------------------------
# Config & Page
# ---------------------------
st.set_page_config(page_title="Smart Assistant | Atlas Copco", layout="wide")
st.markdown("## 🧠 Smart Assistant for Atlas Copco Documents")

# ---------------------------
# Static Product Categories (namespace = your Pinecone namespace)
# ---------------------------
product_categories = [
    {"label": "Mobile Diesel Generators (QAS Series)", "namespace": "altas-copco-qas-manuals"},
    {"label": "Refrigerant Air Dryers (FD Series)",   "namespace": "altas-copco-fd-manuals"},
    {"label": "Portable Compressors (XAS Series)",    "namespace": "altas-copco-xas-manuals"},
    {"label": "Oil-Free Rotary Screw Compressors (Z Series)", "namespace": "altas-copco-z-manuals"},
    {"label": "Scroll Air Compressors (SF Series)",   "namespace": "altas-copco-sf-manuals"},
    {"label": "Oil-Injected Screw Compressors (GA Series)", "namespace": "altas-copco-ga-manuals"},
    {"label": "Desiccant Air Dryers (CD Series)",     "namespace": "altas-copco-cd-manuals"},
]

# ---------------------------
# State
# ---------------------------
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "selected_label" not in st.session_state:
    st.session_state.selected_label = None
if "active_modal" not in st.session_state:
    st.session_state.active_modal = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm_model" not in st.session_state:
    # Change to your preferred model; mini is cheap/fast
    st.session_state.llm_model = "gpt-4o-mini"

# ---------------------------
# Utils
# ---------------------------
def get_unique_files_in_namespace(namespace: str, top_k: int = 100) -> List[str]:
    """Cheap way to list distinct filenames from a namespace by querying a dummy vector."""
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
# Mini "React" Agent Components
# ---------------------------

# 1) RAG tool (wrap your chain so the agent can "call" it)
def make_rag_tool(retriever):
    rag_chain = create_rag_chain(retriever)

    def _rag_tool_fn(query: str, category_namespace: str) -> Dict[str, Any]:
        # Language normalization (same as your direct flow)
        translated = detect_and_translate(query)
        response = rag_chain({
            "input": query,
            "translated_query": translated["translated_query"],
            "language": translated["language"],
            "chat_history": st.session_state.chat_history
        })

        # Normalize evidence
        sources = []
        if response.get("sources"):
            for doc in response["sources"]:
                filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                sources.append({"filename": filename, "page": page})

        return {
            "final_answer": response["answer"],
            "evidence": {"type": "rag_sources", "items": sources},
            "used_tool": "rag",
            "confidence_hint": response.get("confidence", None)
        }

    return _rag_tool_fn

# 2) Fault code tool (from services/fault_code_tool.py)
fault_tool = FaultCodeTool()  # uses default base_dir & mapping inside
parts_tool = PartsTool()
# 3) Router (LLM decides which tool to call)
class RouteDecision(BaseModel):
    tool: Literal["fault_code", "rag", "check_part_inventory"]
    rationale: str

def make_router():
    llm = ChatOpenAI(model=st.session_state.llm_model, temperature=0)
    structured = llm.with_structured_output(RouteDecision)

    def route(query: str) -> RouteDecision:
        instruction = f"""
        You are a router for three tools:
        - Use 'fault_code' if query has a fault code, alarm, error, or issue description.
        - Use 'rag' for general manual/document Q&A.
        - Use 'check_part_inventory' if user asks directly about any part availability OR says 'yes' after fault_code response.
        Always pick exactly one tool.
        User Query: {query}
        """
        return structured.invoke(instruction)
    return route



# 4) Judge (LLM self-check on the result)
class JudgeVerdict(BaseModel):
    ok: bool
    score: float = Field(..., ge=0, le=1, description="0-1 helpfulness/grounding")
    reason: str

def make_judge():
    llm = ChatOpenAI(model=st.session_state.llm_model, temperature=0)
    structured = llm.with_structured_output(JudgeVerdict)

    def judge(query: str, candidate_answer: str, evidence: Dict[str, Any]) -> JudgeVerdict:
        # Create a string-based prompt instead of passing a dict
        evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)

        full_prompt = f"""
            You are a strict answer judge.

            **Task**: Check if the answer is grounded in the provided evidence and sufficiently addresses the user's query.

            **User Query:**
            {query}

            **Candidate Answer:**
            {candidate_answer}

            **Evidence:**
            {evidence_json}

            **Rules:**
            - If evidence.type == 'rag_sources', ensure the answer could reasonably come from those documents.
            - If evidence.type == 'fault_matches', ensure the answer matches the selected fault JSON entries.
            - Return ok=true only if the answer is correct, non-hallucinated, and helpful.
            - Score: 0 (bad) to 1 (excellent).

            Return a structured object matching the schema.
            """

        return structured.invoke(full_prompt)

    return judge


router = make_router()
judge_answer = make_judge()

# ---------------------------
# UI Tabs
# ---------------------------
tab1, tab2 = st.tabs(["💬 Ask Question", "📁 View PDFs"])

# ---------------------------
# Tab 1: Ask Question (Agent flow)
# ---------------------------
with tab1:
    st.subheader("Select a Product Category to Ask Questions")

    cols = st.columns(len(product_categories))
    for i, item in enumerate(product_categories):
        label = item["label"]
        namespace = item["namespace"]
        is_selected = (st.session_state.selected_namespace == namespace)
        border_color = "#4da6ff" if is_selected else "#ccc"
        with cols[i]:
            if st.button(f"📦 {label}", key=f"ask_card_{namespace}"):
                st.session_state.selected_namespace = namespace
                st.session_state.selected_label = label

            st.markdown(f"""
                <style>
                div[data-testid="column"] > div:has(button[kind='secondary'][key='ask_card_{namespace}']) {{
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                }}
                </style>
            """, unsafe_allow_html=True)

    selected_ns = st.session_state.selected_namespace
    selected_label = st.session_state.selected_label

    if selected_ns:
        # Build a fresh RAG tool for the selected namespace
        retriever = PineconeRetrieverWithThreshold(namespace=selected_ns)
        rag_tool_fn = make_rag_tool(retriever)

        query = st.text_input(f"Ask something in `{selected_label}`:", key="user_input")

        if query:
            with st.spinner("Thinking..."):
                # ====== React-ish flow: Route -> Act -> Observe -> Judge -> (optional) Fallback ======
                decision = router(query)
                print("\n========================")
                print("🔹 Query received:", query)
                print("🔹 Router decision output:", decision)
                print("========================\n")

                used_tool = decision.tool

                if used_tool == "fault_code":
                    tool_input = {
                        "query": query,
                        "category_namespace": selected_ns
                    }
                    result = fault_tool.run(tool_input)
                elif used_tool == "check_part_inventory":
                    last_bot = st.session_state.chat_history[-1] if st.session_state.chat_history else ""
                    

                    prompt = f"""
                    You are an assistant. Extract ONLY the part name that should be checked in inventory
                    from this text:
                    ---
                    {last_bot}
                    ---
                    Reply with only the part name, nothing else.
                    """
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    candidate_part = llm.invoke(prompt).content.strip()
                    print("Candidate " , candidate_part)
                    if not candidate_part:
                        candidate_part = query  

                    part_res = parts_tool.run({"part_name": candidate_part})

                    result = {
                        "final_answer": part_res, 
                        "evidence": {"type": "inventory", "items": []},
                        "used_tool": "check_part_inventory"
                    }


                else:
                    result = rag_tool_fn(query=query, category_namespace=selected_ns)
                verdict = judge_answer(query, result["final_answer"], result["evidence"])

                # Optional: simple fallback — if judge says not ok, try the *other* tool once
                if not verdict.ok:
                    fallback_tool = "rag" if used_tool == "fault_code" else "fault_code"
                    if fallback_tool == "fault_code":
                        alt = fault_tool.run({
                            "query": query,
                            "category_namespace": selected_ns
                        })

                    else:
                        alt = rag_tool_fn(query=query, category_namespace=selected_ns)

                    alt_verdict = judge_answer(query, alt["final_answer"], alt["evidence"])
                    # Pick the better scored one
                    if alt_verdict.score > verdict.score:
                        result, verdict, used_tool = alt, alt_verdict, fallback_tool

            # Persist chat
            st.session_state.chat_history.append(f"User: {query}")
            st.session_state.chat_history.append(f"Bot: {result['final_answer']}")

            # ---- Render
            st.subheader("Answer:")
            st.write(result["final_answer"])

            # Tool / Evidence Badges
            tool_badge = "🔧 Fault Code JSON" if used_tool == "fault_code" else "📄 RAG (Manuals)"
            st.markdown(f"**Source Tool:** {tool_badge}")

            # Show evidence
            if result.get("evidence"):
                ev = result["evidence"]
                if ev["type"] == "rag_sources" and ev["items"]:
                    st.markdown("#### 📚 Relevant Sources:")
                    for s in ev["items"]:
                        st.markdown(f"- {s['filename']} (page {s['page']})")

            # Judge verdict
            st.markdown("---")
            ok_emoji = "✅" if verdict.ok else "⚠️"
            st.markdown(f"**Validation:** {ok_emoji} score={verdict.score:.2f} — {verdict.reason}")

            # Chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("🕘 Chat History")
                for entry in st.session_state.chat_history:
                    st.markdown(f"- {entry}")
    else:
        st.info("Please select a product category to begin.")

# ---------------------------
# Tab 2: View PDFs
# ---------------------------
with tab2:
    st.subheader("View PDFs in a Product Category")

    cols = st.columns(len(product_categories))
    for i, item in enumerate(product_categories):
        label = item["label"]
        namespace = item["namespace"]
        is_selected = (st.session_state.active_modal == namespace)
        border_color = "#4da6ff" if is_selected else "#ccc"
        with cols[i]:
            if st.button(f"📦 {label}", key=f"view_card_{namespace}"):
                st.session_state.active_modal = namespace

            st.markdown(f"""
                <style>
                div[data-testid="column"] > div:has(button[kind='secondary'][key='view_card_{namespace}']) {{
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
