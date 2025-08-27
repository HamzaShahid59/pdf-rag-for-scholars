import json
import random
from pymongo import MongoClient

# -----------------------------
# MongoDB Connection
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["test_assistant_inventory"]  # Replace with your DB name
parts_collection = db["parts"]  # Collection name

# -----------------------------
# Load your JSON data
# -----------------------------
json_data_1 = [
    {
        "code": "FX-PDP-HI",
        "name": "High Pressure Dew Point",
        "parts": [
            {"sku": "FAN-ASSY-FD", "quantity": 1},
            {"sku": "FILTER-LIQ-REF", "quantity": 1}
        ]
    },
    {
        "code": "FX-PDP-LO",
        "name": "Low Pressure Dew Point",
        "parts": []
    },
    {
        "code": "FX-FRZ-L2",
        "name": "Freeze Protection (L2)",
        "parts": [
            {"sku": "VALVE-HGBV", "quantity": 1},
            {"sku": "SNS-TEMP-PT1000", "quantity": 1}
        ]
    },
    {
        "code": "FX-PROBE",
        "name": "Probe Failure",
        "parts": [
            {"sku": "SNS-TEMP-PT1000", "quantity": 1}
        ]
    },
    {
        "code": "FX-EE",
        "name": "EEPROM Error",
        "parts": [
            {"sku": "CTRL-ELEK", "quantity": 1}
        ]
    },
    {
        "code": "FX-FAN",
        "name": "Fan Failure",
        "parts": [
            {"sku": "FAN-ASSY-FD", "quantity": 1},
            {"sku": "RELAY-HP-LP", "quantity": 1}
        ]
    },
    {
        "code": "FX-HP-SW",
        "name": "High Pressure Switch Trip",
        "parts": [
            {"sku": "RELAY-HP-LP", "quantity": 1},
            {"sku": "FAN-ASSY-FD", "quantity": 1}
        ]
    },
    {
        "code": "FX-COMP-TH",
        "name": "Compressor Discharge Temp High",
        "parts": [
            {"sku": "FILTER-LIQ-REF", "quantity": 1},
            {"sku": "COMP-REF-1PH", "quantity": 1}
        ]
    },
    {
        "Code": "FD-PDP-HI",
        "PartsCommonlyUsed": [
            "Refrigerant (R134a/R404A/R410A per model)",
            "Condenser fan",
            "Fan pressure switch"
        ]
    },
    {
        "Code": "FD-FRZ",
        "PartsCommonlyUsed": [
            "Hot-gas bypass valve",
            "Fan switch"
        ]
    },
    {
        "Code": "FD-COND-HI",
        "PartsCommonlyUsed": [
            "Fan motor",
            "Fan switch"
        ]
    },
    {
        "Code": "FD-COND-LO",
        "PartsCommonlyUsed": [
            "Fan switch"
        ]
    },
    {
        "Code": "FD-COMP-STOP",
        "PartsCommonlyUsed": [
            "Capacitor/overload",
            "Contactor"
        ]
    },
    {
        "Code": "FD-DRAIN-NOOP",
        "PartsCommonlyUsed": [
            "Electronic drain kit"
        ]
    },
    {
        "Code": "FD-DRAIN-LEAK",
        "PartsCommonlyUsed": [
            "Electronic drain"
        ]
    },
    {
        "Code": "FD-DP-SENSOR",
        "PartsCommonlyUsed": [
            "PDP probe"
        ]
    }
]

json_data_2 = [
    {
        "FaultCode": "LOP-01",
        "PartsRequired": [
            "Engine oil",
            "Oil filter",
            "Pressure sensor"
        ]
    },
    {
        "FaultCode": "HCT-02",
        "PartsRequired": [
            "Coolant",
            "Thermostat",
            "Radiator/fan parts"
        ]
    },
    {
        "FaultCode": "ALT-03",
        "PartsRequired": [
            "Alternator",
            "V-belt",
            "Regulator"
        ]
    },
    {
        "FaultCode": "LFL-04",
        "PartsRequired": [
            "Diesel fuel",
            "Sensor (if faulty)"
        ]
    },
    {
        "FaultCode": "LCL-05",
        "PartsRequired": [
            "Coolant",
            "Sensor",
            "Hoses"
        ]
    },
    {
        "FaultCode": "GOV-06",
        "PartsRequired": [
            "AVR (Automatic Voltage Regulator)"
        ]
    },
    {
        "FaultCode": "GUV-07",
        "PartsRequired": [
            "AVR",
            "Alternator repair kit"
        ]
    },
    {
        "FaultCode": "GOF-08",
        "PartsRequired": [
            "Governor actuator"
        ]
    }
]

# -----------------------------
# Collect All Unique Parts
# -----------------------------
all_parts = set()

# From first JSON
for item in json_data_1:
    if "parts" in item and item["parts"]:
        for p in item["parts"]:
            all_parts.add(p["sku"])
    if "PartsCommonlyUsed" in item:
        for p in item["PartsCommonlyUsed"]:
            all_parts.add(p)

# From second JSON
for item in json_data_2:
    if "PartsRequired" in item:
        for p in item["PartsRequired"]:
            all_parts.add(p)

# -----------------------------
# Generate Dummy Data for Each Part
# -----------------------------
dummy_parts = []
for part in all_parts:
    dummy_parts.append({
        "name": part,
        "stock_quantity": random.randint(10, 200),
        "price": round(random.uniform(50, 1500), 2),
        "delivery_time": f"{random.randint(2, 10)} days",
        "manufacturing_time": f"{random.randint(5, 20)} days"
    })

# -----------------------------
# Insert into MongoDB
# -----------------------------
if dummy_parts:
    parts_collection.insert_many(dummy_parts)
    print(f"✅ Inserted {len(dummy_parts)} dummy parts into MongoDB.")
else:
    print("⚠ No parts found to insert.")
