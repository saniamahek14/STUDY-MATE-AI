# test_ollama.py
# Run this FIRST. It calls Ollama's local REST API (no internet needed).

import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "Explain a transistor in 2 simple lines."}],
        "stream": False,
    },
    timeout=300,
)
response.raise_for_status()
print(response.json()["message"]["content"])
