# 📚 StudyMate AI

A local AI-powered multimodal study assistant. Runs on your own computer with
a local LLM (Ollama), local RAG (ChromaDB) and local voice (Whisper).

## Features
- Upload a PDF of notes and ask questions (RAG)
- Summary, key points, MCQs, flashcards, simple explanations
- Conversation memory (follow-up questions work)
- Voice input (speak your question) and voice output (read answers aloud)
- Lecture audio -> text -> summary
- Answer modes: Notes only, General knowledge, Web search (optional), Auto

## Tech
Python 3.12, Streamlit, Ollama REST API (llama3.2:3b), sentence-transformers,
ChromaDB, pypdf, faster-whisper, pyttsx3

## How it works
PDF -> chunks -> embeddings -> ChromaDB -> relevant chunks + question -> local LLM -> answer

## Setup
1. Install Python 3.12 and Ollama (https://ollama.com)
2. `ollama pull llama3.2:3b`
3. Run:

```
git clone https://github.com/YOUR-USERNAME/studymate-ai.git
cd studymate-ai
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Project structure
| Path | Purpose |
|---|---|
| app.py | Streamlit website |
| core/llm.py | Calls Ollama's local REST API |
| core/prompts.py | System prompt and task prompts |
| core/rag.py | PDF reading, chunking, embeddings, ChromaDB |
| core/voice.py | Speech-to-text and text-to-speech |
| core/web.py | Optional web search |
| test_ollama.py | First local AI test |

## Limits
- Small local models can make mistakes; check the source under each answer.
- Scanned PDFs (images) have no text and won't work.
- Web search mode needs internet; everything else works offline.
- Voice output uses your computer's built-in voice.
