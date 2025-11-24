import requests

r = requests.post("http://127.0.0.1:3000/chat",
    json={"message": "what is leg numbness?", "lang": "en"}
)

print("STATUS:", r.status_code)
print("BODY:", r.text)
