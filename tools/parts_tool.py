# tools/parts_tool.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pymongo import MongoClient
from pymongo.collation import Collation
from dotenv import load_dotenv
import os
import re

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment. Check your .env file.")

# Collation: strength=2 => case-insensitive, diacritics-insensitive (no regex needed)
CASE_INSENSITIVE = Collation(locale="en", strength=2)

# Common leading noise phrases to strip from user text
STOP_PHRASES = [
    "availability of", "availability", "availabilty of", "availabilty",
    "check for", "check the", "check", "in stock", "stock of",
    "do you have", "can you check", "please check", "ok check", "yes check",
]

def _normalize_spaces(s: str) -> str:
    return " ".join(s.split())

def _clean_query(raw: str) -> str:
    q = raw.strip().lower()
    # remove any stop phrase at the start
    for p in STOP_PHRASES:
        if q.startswith(p + " "):
            q = q[len(p) + 1 : ]
    # collapse spaces
    q = _normalize_spaces(q)
    return q

def _tokens(s: str) -> List[str]:
    # letters/digits only tokens, lowercased
    return re.findall(r"[a-z0-9]+", s.lower())

def _haystack_from_doc(doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    name = doc.get("name")
    if isinstance(name, str):
        parts.append(name.lower())
    aliases = doc.get("aliases") or []
    if isinstance(aliases, list):
        parts.extend([str(a).lower() for a in aliases if isinstance(a, (str, bytes))])
    return " ".join(_normalize_spaces(p) for p in parts if p)

@dataclass
class PartsTool:
    uri: str = MONGO_URI
    db: str = "inventory_db"
    coll: str = "parts"

    def __post_init__(self):
        self.client = MongoClient(self.uri)
        self.collection = self.client[self.db][self.coll]

    name: str = "check_part_inventory"
    description: str = "Check if a specific spare part is available in inventory. Input should include the part name."

    def run(self, tool_input: Dict[str, Any]) -> str:
        """Fetch part details by name from MongoDB without regex.
        Flow:
          1) Clean NL query to a candidate part name.
          2) Case-insensitive EXACT match on 'name' via collation.
          3) Case-insensitive EXACT match on 'aliases' via collation.
          4) Token-AND fallback in Python (no regex): all words must appear in name or aliases.
        """
        raw = tool_input.get("part_name")
        if not raw or not isinstance(raw, str):
            return "⚠️ No part name provided."

        candidate = _clean_query(raw)
        if not candidate:
            return "⚠️ Please provide a valid part name."

        # ---------- 1) Case-insensitive exact match on name ----------
        doc = self.collection.find_one({"name": candidate}, collation=CASE_INSENSITIVE)

        # ---------- 2) Case-insensitive exact match on aliases ----------
        if not doc:
            # Equality on array elements works; collation applies to the whole query
            doc = self.collection.find_one({"aliases": candidate}, collation=CASE_INSENSITIVE)

        # ---------- 3) Token-AND fallback (no regex) ----------
        if not doc:
            toks = _tokens(candidate)
            if toks:
                # fetch a reasonable number of docs (adjust as needed)
                cursor = self.collection.find(
                    {},  # narrow this if your collection is large
                    projection={"name": 1, "aliases": 1, "price": 1, "stock_quantity": 1, "delivery_time": 1, "manufacturing_time": 1},
                ).limit(1000)

                best_doc: Optional[Dict[str, Any]] = None
                best_score = -1

                for d in cursor:
                    hay = _haystack_from_doc(d)
                    # simple contains: require all tokens to appear
                    if all(t in hay for t in toks):
                        # prefer the one with most token hits (tie-breaker)
                        score = sum(hay.count(t) for t in toks)
                        if score > best_score:
                            best_doc = d
                            best_score = score

                doc = best_doc

        if not doc:
            return f"❌ Part '{raw}' not found in inventory."

        # ---------- Render result ----------
        name = doc.get("name", candidate)
        price = doc.get("price", "N/A")
        qty = doc.get("stock_quantity", 0)
        delivery = doc.get("delivery_time", "N/A")
        mfg_time = doc.get("manufacturing_time", "N/A")

        try:
            in_stock = float(qty) > 0
        except Exception:
            in_stock = False

        if in_stock:
            return (
                f"✅ Part '{name}' is in stock.\n"
                f"Quantity: {qty}\n"
                f"Price: ${price}\n"
                f"Delivery: {delivery}"
            )
        else:
            return (
                f"⚠️ Part '{name}' is NOT in stock.\n"
                f"Manufacturing time: {mfg_time}\n"
                f"Price: ${price}"
            )
