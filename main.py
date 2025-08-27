from dotenv import load_dotenv
load_dotenv()

from typing import List, Literal, Optional, Dict, Any
import re
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
st.set_page_config(page_title="OptimAIze Assitant", layout="wide")
st.markdown("## 🧠 OptimAIze Assitant")

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
    # store as list of {role: "user"|"assistant", "content": str}
    st.session_state.chat_history = []
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o-mini"
# NEW: conversation memory for routing
if "awaiting_inventory" not in st.session_state:
    st.session_state.awaiting_inventory = False
if "last_bot_text" not in st.session_state:
    st.session_state.last_bot_text = ""

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

# --- Helpers for routing / extraction ---
INVENTORY_TRIGGERS = [
    "check", "inventory", "availability", "available", "in stock",
    "stock", "do you have", "can you check", "please check", "ok check",
    "yes check", "check for", "check the", "check part"
]

def looks_like_inventory_request(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in INVENTORY_TRIGGERS)

def extract_part_from_user(query: str) -> Optional[str]:
    """
    Try to pull the part name from the user's message with simple patterns.
    Examples it catches:
    - "check for the fan motor"
    - "please check fan motor"
    - "check fan switch"
    - "is fan motor in stock?"
    Fallback to None; LLM extractor will still run later.
    """
    q = query.lower()
    # simple shelves for known part nouns; extend if needed
    known_heads = ["fan motor", "fan switch", "switch", "motor", "filter", "belt"]
    for head in known_heads:
        if head in q:
            return head
    # generic patterns
    m = re.search(r"(?:check(?:\s+for)?|availability of|in stock|stock of)\s+(the\s+)?([a-z0-9\- ]+)", q)
    if m:
        return m.group(2).strip()
    return None

# ---------------------------
# Mini "React" Agent Components
# ---------------------------

# 1) RAG tool
def make_rag_tool(retriever):
    rag_chain = create_rag_chain(retriever)

    def _rag_tool_fn(query: str, category_namespace: str) -> Dict[str, Any]:
        translated = detect_and_translate(query)
        response = rag_chain({
            "input": query,
            "translated_query": translated["translated_query"],
            "language": translated["language"],
            "chat_history": [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history]
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

# 2) Tools
fault_tool = FaultCodeTool()
parts_tool = PartsTool()

# 3) Router
class RouteDecision(BaseModel):
    tool: Literal["fault_code", "rag", "check_part_inventory"]
    rationale: str

def make_router():
    llm = ChatOpenAI(model=st.session_state.llm_model, temperature=0)
    structured = llm.with_structured_output(RouteDecision)

    def route(query: str) -> RouteDecision:
        # ---- Pre-route heuristic (deterministic) ----
        # If the user asks to check inventory or we are awaiting inventory and user confirms,
        # force check_part_inventory to avoid misrouting to fault_code.
        awaiting = st.session_state.awaiting_inventory
        last_bot = st.session_state.last_bot_text
        if looks_like_inventory_request(query) or (awaiting and ("ok" in query.lower() or "yes" in query.lower() or "check" in query.lower())):
            # print("DEBUG: Heuristic routed to check_part_inventory")  # debug
            return RouteDecision(tool="check_part_inventory", rationale="Heuristic: inventory intent recognized")

        # ---- LLM router with context-aware instruction ----
        instruction = f"""
        You are a router for three tools:
        - Use 'fault_code' if the user describes a fault code, alarm, error, or issue description seeking diagnostics.
        - Use 'rag' for general manual/document Q&A and requests not about part availability.
        - Use 'check_part_inventory' if the user asks to check availability/stock/inventory of a part,
          OR if the last assistant message listed parts or asked to check inventory and the user confirms with short phrases
          like "ok", "yes", "please check", "check fan motor", "check the fan motor", etc.

        Conversation signals:
        - awaiting_inventory={awaiting}
        - last_assistant_message:
        ---
        {last_bot}
        ---

        If the user message includes any inventory/availability trigger words:
        {", ".join(INVENTORY_TRIGGERS)}
        OR mentions a part name after such a trigger, choose 'check_part_inventory' even if the message also describes a fault.

        Always pick exactly one tool.

        User Query: {query}
        """
        return structured.invoke(instruction)
    return route

# 4) Judge
class JudgeVerdict(BaseModel):
    ok: bool
    score: float = Field(..., ge=0, le=1, description="0-1 helpfulness/grounding")
    reason: str

def make_judge():
    llm = ChatOpenAI(model=st.session_state.llm_model, temperature=0)
    structured = llm.with_structured_output(JudgeVerdict)

    def judge(query: str, candidate_answer: str, evidence: Dict[str, Any]) -> JudgeVerdict:
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
# Sidebar Navigator
# ---------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    nav = st.radio("Go to", options=["💬 Ask Question", "📁 View PDFs"], label_visibility="collapsed")

# ---------------------------
# Category Selection (Top - stays where it is)
# ---------------------------
st.subheader("Select a Product Category")
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
            # print("DEBUG: Selected namespace ->", namespace)  # debug
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

# ---------------------------
# Page: Ask Question (Chat-first UI)
# ---------------------------
if nav == "💬 Ask Question":
    if not selected_ns:
        st.info("Please select a product category to begin.")
    else:
        # Build RAG tool for the selected namespace
        retriever = PineconeRetrieverWithThreshold(namespace=selected_ns)
        rag_tool_fn = make_rag_tool(retriever)

        # --- CHAT HISTORY (top) ---
        if st.session_state.chat_history:
            st.subheader(f"💬 Chat — {selected_label}")
            for msg in st.session_state.chat_history:
                # Backwards compatibility if older string style exists
                if isinstance(msg, str):
                    if msg.startswith("User: "):
                        role, content = "user", msg[6:]
                    elif msg.startswith("Bot: "):
                        role, content = "assistant", msg[5:]
                    else:
                        role, content = "assistant", msg
                else:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")

                with st.chat_message(role):
                    st.markdown(content)

        # --- CHAT INPUT (bottom) ---
        user_query = st.chat_input(f"Ask something in `{selected_label}`")
        if user_query:
            # 1) Immediately show user's message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            # print("DEBUG: Appended user message")  # debug

            # 2) Assistant placeholder + spinner
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    decision = router(user_query)
                    used_tool = decision.tool
                    # print("DEBUG: Router decision ->", used_tool)  # debug

                    if used_tool == "fault_code":
                        tool_input = {"query": user_query, "category_namespace": selected_ns}
                        result = fault_tool.run(tool_input)

                    elif used_tool == "check_part_inventory":
                        # Try extracting from user text first
                        candidate_part = extract_part_from_user(user_query)

                        last_bot_text = st.session_state.last_bot_text
                        if not candidate_part:
                            # Use LLM extraction with both user text AND last assistant
                            prompt = f"""
                            You are an assistant. Extract ONLY the part name that should be checked in inventory
                            from the following texts. Prefer the user's message; if not explicit, use the assistant context.
                            ---
                            USER: {user_query}
                            ASSISTANT_CONTEXT: {last_bot_text}
                            ---
                            Reply with only the part name, nothing else.
                            """
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                            candidate_part = llm.invoke(prompt).content.strip()

                        if not candidate_part:
                            candidate_part = "fan motor"  # safe fallback for short confirmations

                        # print("DEBUG: Candidate part ->", candidate_part)  # debug
                        part_res = parts_tool.run({"part_name": candidate_part})
                        result = {
                            "final_answer": part_res,
                            "evidence": {"type": "inventory", "items": []},
                            "used_tool": "check_part_inventory"
                        }

                    else:
                        result = rag_tool_fn(query=user_query, category_namespace=selected_ns)

                    verdict = judge_answer(user_query, result["final_answer"], result.get("evidence", {}))

                    # Optional fallback
                    if not verdict.ok and used_tool in ("fault_code", "rag"):
                        fallback_tool = "rag" if used_tool == "fault_code" else "fault_code"
                        if fallback_tool == "fault_code":
                            alt = fault_tool.run({"query": user_query, "category_namespace": selected_ns})
                        else:
                            alt = rag_tool_fn(query=user_query, category_namespace=selected_ns)

                        alt_verdict = judge_answer(user_query, alt["final_answer"], alt.get("evidence", {}))
                        if alt_verdict.score > verdict.score:
                            result, verdict, used_tool = alt, alt_verdict, fallback_tool
                            # print("DEBUG: Used fallback tool ->", fallback_tool)  # debug

                # 3) Fill assistant bubble content
                st.markdown(result["final_answer"])

                # Tool / Evidence info under the reply
                tool_badge = (
                    "🔧 Fault Codes Data" if used_tool == "fault_code"
                    else ("🧾 Inventory" if used_tool == "check_part_inventory" else "📄 (Manuals)")
                )
                st.caption(f"**Source Tool:** {tool_badge}")

                ev = result.get("evidence")
                if ev and ev.get("type") == "rag_sources" and ev.get("items"):
                    with st.expander("📚 Relevant Sources"):
                        for s in ev["items"]:
                            st.markdown(f"- {s['filename']} (page {s['page']})")

                # Keep validation hidden in history; place it below the reply if you later re-enable
                # ok_emoji = "✅" if verdict.ok else "⚠️"
                # st.markdown("---")
                # st.markdown(f"**Validation:** {ok_emoji} score={verdict.score:.2f} — {verdict.reason}")

            # 4) Persist assistant message and update convo memory
            st.session_state.chat_history.append({"role": "assistant", "content": result["final_answer"]})
            st.session_state.last_bot_text = result["final_answer"]

            # If the assistant just listed parts (common pattern from fault tool), set awaiting_inventory
            if used_tool == "fault_code" and ("Parts Commonly Used" in result["final_answer"] or "Please let me know if you’d like me to check" in result["final_answer"]):
                st.session_state.awaiting_inventory = True
                # print("DEBUG: awaiting_inventory set -> True")  # debug
            elif used_tool == "check_part_inventory":
                # Once we check inventory, we can turn it off (until next fault diagnosis)
                st.session_state.awaiting_inventory = False
                # print("DEBUG: awaiting_inventory set -> False")  # debug

# ---------------------------
# Page: View PDFs (via navigator)
# ---------------------------
elif nav == "📁 View PDFs":
    if not selected_ns:
        st.info("Please select a product category to view PDFs.")
    else:
        st.subheader(f"View PDFs in `{selected_ns}`")
        files = get_unique_files_in_namespace(selected_ns)
        with st.expander(f"📁 Files in `{selected_ns}`", expanded=True):
            if files:
                for file in files:
                    st.markdown(f"- {file}")
            else:
                st.info("No PDFs found in this namespace.")
