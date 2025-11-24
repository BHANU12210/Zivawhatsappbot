# openai_sdk_debug.py
import os, traceback, time
from openai import OpenAI
from openai.error import APIConnectionError

KEY = os.getenv("OPENAI_API_KEY")
print("ENV KEY present?", bool(KEY))

if not KEY:
    raise SystemExit("OPENAI_API_KEY missing in env")

client = OpenAI(api_key=KEY)

def try_once():
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":"Ping: confirm connectivity and return a short line."}],
            max_tokens=10,
            timeout=20
        )
        print("SDK RAW RESP:", resp)
        # attempt robust extraction
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
        return True
    except APIConnectionError as e:
        print("APIConnectionError:", repr(e))
        traceback.print_exc()
        return False
    except Exception as e:
        print("OTHER ERROR:", repr(e))
        traceback.print_exc()
        return False

if __name__ == "__main__":
    for attempt in range(1,4):
        print(f"Attempt {attempt} -- {time.strftime('%X')}")
        ok = try_once()
        if ok:
            break
        time.sleep(2)
    else:
        print("All attempts failed.")
