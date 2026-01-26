import json
import os
import urllib.request

BASE_URL = os.getenv("LT_BASE_URL", "http://localhost:8080")
TOKEN = os.getenv("LT_AUTH_TOKEN", "").strip()

payload = {"q": "Hello", "source": "en", "target": "hi"}
headers = {"Content-Type": "application/json"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"

req = urllib.request.Request(
    f"{BASE_URL}/translate",
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST",
)

with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode("utf-8"))
