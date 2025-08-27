# tools/parts_tool.py
from pymongo import MongoClient
from typing import Dict, Any

class PartsTool:
    name = "check_part_inventory"
    description = "Check if a specific spare part is available in inventory. Input should include the part name."

    def __init__(self, uri="mongodb://localhost:27017/", db="test_assistant_inventory", coll="parts"):
        self.client = MongoClient(uri)
        self.collection = self.client[db][coll]

    def run(self, tool_input: Dict[str, Any]) -> str:
        """Fetch part details by name from MongoDB."""
        part_name = tool_input.get("part_name")
        if not part_name:
            return "⚠️ No part name provided."

        doc = self.collection.find_one({"name": {"$regex": f"^{part_name}$", "$options": "i"}})
        if not doc:
            return f"❌ Part '{part_name}' not found in inventory."

        if doc["stock_quantity"] > 0:
            return (
                f"✅ Part '{doc['name']}' is in stock.\n"
                f"Quantity: {doc['stock_quantity']}\n"
                f"Price: ${doc['price']}\n"
                f"Delivery: {doc['delivery_time']}"
            )
        else:
            return (
                f"⚠️ Part '{doc['name']}' is NOT in stock.\n"
                f"Manufacturing time: {doc['manufacturing_time']}\n"
                f"Price: ${doc['price']}"
            )
