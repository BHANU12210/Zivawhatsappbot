# test_openai_direct.py
from openai import OpenAI
import os, sys

KEY = os.getenv("OPENAI_API_KEY")
print("ENV KEY present?", bool(KEY))
if not KEY:
    print("ERROR: OPENAI_API_KEY missing in environment.")
    sys.exit(1)

client = OpenAI(api_key=KEY)
try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":"Explain leg numbness in one sentence."}],
        max_tokens=60
    )
    # robust extraction
    ans = None
    if hasattr(resp, "choices") and len(resp.choices) > 0:
        ch = resp.choices[0]
        if isinstance(ch, dict):
            ans = ch.get("message", {}).get("content")
        else:
            msg = getattr(ch, "message", None)
            if msg:
                ans = getattr(msg, "content", None)
    if not ans and hasattr(resp, "output_text"):
        ans = getattr(resp, "output_text")
    print("ANSWER:", ans)
except Exception as e:
    print("ERROR:", repr(e))
