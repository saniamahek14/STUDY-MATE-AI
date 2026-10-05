# core/llm.py
# Talks to the local Ollama server through its REST API (localhost:11434).

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"  # change to "phi3" if you pulled that instead


def chat(messages, temperature=0.3):
    """
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    temperature: low (0.1-0.3) = focused, high (0.8+) = creative
    Returns the model's reply as a string.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"]
