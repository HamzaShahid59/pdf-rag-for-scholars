from langchain_openai import ChatOpenAI
import json, os
from langchain.tools import BaseTool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAMESPACE_TO_FAULT_JSON = {
    "altas-copco-qas-manuals": os.path.join(BASE_DIR, "fault_codes", "qas.json"),
    "altas-copco-fd-manuals":  os.path.join(BASE_DIR, "fault_codes", "fd.json"),
    "altas-copco-xas-manuals": os.path.join(BASE_DIR, "fault_codes", "xas.json"),
    "altas-copco-z-manuals":   os.path.join(BASE_DIR, "fault_codes", "z.json"),
    "altas-copco-sf-manuals":  os.path.join(BASE_DIR, "fault_codes", "sf.json"),
    "altas-copco-ga-manuals":  os.path.join(BASE_DIR, "fault_codes", "ga.json"),
    "altas-copco-cd-manuals":  os.path.join(BASE_DIR, "fault_codes", "cd.json"),
}

class FaultCodeTool(BaseTool):
    name: str = "fault_code_tool"
    description: str = (
        "Look up fault/alarms/errors in a product-specific JSON manual. "
        "Given a namespace and a query (fault code or description), "
        "the tool retrieves the JSON and lets the LLM find the relevant entry "
        "and explain it with as much detail as possible."
    )

    def _run(self, query: str, category_namespace: str) -> dict:
        try:
            if not query or not category_namespace:
                return {
                    "final_answer": "❌ Missing required fields: 'query' and 'category_namespace'",
                    "evidence": {"type": "fault_matches", "items": []},
                    "used_tool": "fault_code"
                }

            json_path = NAMESPACE_TO_FAULT_JSON.get(category_namespace)
            if not json_path or not os.path.exists(json_path):
                return {
                    "final_answer": f"❌ No JSON file found for namespace '{category_namespace}'",
                    "evidence": {"type": "fault_matches", "items": []},
                    "used_tool": "fault_code"
                }

            # Load full fault dataset
            with open(json_path, "r", encoding="utf-8") as f:
                fault_data = json.load(f)

            # LLM reasoning step
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            system_prompt = (
                "You are a fault code expert. You are given:\n"
                "1. A user query (could be a fault code like FD-DRAIN-NOOP, "
                "or a description like 'Condensate trap inoperative').\n"
                "2. A JSON dataset of all possible fault entries for this machine.\n\n"
                "Each entry has fields like Code, ManualTermCondition, LikelyRootCauses, FixSteps, PartsCommonlyUsed.\n\n"
                "Your job:\n"
                "- Find the most relevant entry/entries for the query.\n"
                "- Explain the issue in detail: what the fault means, likely root causes, recommended fix steps, and parts involved.\n"
                "- If nothing matches, explicitly say 'No matching fault found'.\n\n"
                "IMPORTANT:\n"
                "- If any parts are commonly used in the fix, mention them clearly.\n"
                "- At the end, if parts are mentioned, ALWAYS append this EXACT sentence:\n"
                "'If you’d like me to check the availability of any of these parts in the inventory, please let me know.'"
            )

            user_prompt = f"""
            Query: {query}

            Fault JSON Dataset:
            {json.dumps(fault_data, ensure_ascii=False, indent=2)}
            """

            llm_response = llm.invoke([{"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user_prompt}])

            return {
                "final_answer": llm_response.content.strip(),
                "evidence": {"type": "fault_matches", "items": fault_data},
                "used_tool": "fault_code"
            }

        except Exception as e:
            return {
                "final_answer": f"⚠️ Error while processing fault codes: {str(e)}",
                "evidence": {"type": "fault_matches", "items": []},
                "used_tool": "fault_code"
            }

    async def _arun(self, query: str, category_namespace: str) -> dict:
        return self._run(query, category_namespace)
