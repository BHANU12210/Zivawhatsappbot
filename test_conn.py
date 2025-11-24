# test_conn.py
import os
import requests

key = os.getenv("OPENAI_API_KEY")
print("KEY present?", bool(key))
try:
    r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    print("STATUS:", r.status_code)
    print("TEXT:", r.text[:1000])
except Exception as e:
    print("ERROR:", repr(e))
