import csv
import os
import httpx
from collections import defaultdict
from datetime import datetime, timezone

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "POS_sample_transactions.csv")
API_URL = "http://localhost:8000/pos/ingest"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
        
    groups = defaultdict(float)
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Group by unique transaction identifiers
            key = (row["store_id"], row["order_date"], row["order_time"])
            groups[key] += float(row["total_amount"])
            
    transactions = []
    
    for i, (key, basket_value) in enumerate(groups.items()):
        store_id, o_date, o_time = key
        
        # order_date format in CSV is "10-04-2026" -> parse as DD-MM-YYYY
        dt = datetime.strptime(f"{o_date} {o_time}", "%d-%m-%Y %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        timestamp_iso = dt.isoformat()
        
        txn_id = f"TXN_{o_date}_{o_time}_{store_id}_{i}"
        
        transactions.append({
            "transaction_id": txn_id,
            "store_id": store_id,
            "timestamp": timestamp_iso,
            "basket_value_inr": basket_value
        })
        
    if not transactions:
        print("No transactions to process.")
        return
        
    payload = {"transactions": transactions}
    
    print(f"Sending {len(transactions)} POS transactions to API...")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(API_URL, json=payload)
            resp.raise_for_status()
            print("API Response:", resp.json())
    except Exception as e:
        print(f"Error posting to API: {e}")

if __name__ == "__main__":
    main()
