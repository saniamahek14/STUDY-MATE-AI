# app.py
# StudyMate AI - Streamlit website.
# Run with:  streamlit run app.py

import os
import tempfile

import streamlit as st

from core import rag, voice, web
from core.llm import chat
from core.prompts import SYSTEM_PROMPT, TASKS, build_prompt

st.set_page_config(page_title="StudyMate AI", page_icon="📚")
st.title("📚 StudyMate AI")
st.caption("Your local AI study assistant: notes, voice, quizzes")

# ---------- Memory (session_state survives reruns) ----------
defaults = {"messages": [], "doc_name": None, "transcript": "", "last_voice": None}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def save_temp(data, suffix):
    """Write bytes to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def get_context(question, mode):
    """Decide where the answer's information comes from.
    Returns (context, source_label, links, use_context)."""
    if mode == "General knowledge":
        return "", "AI's own knowledge", [], False
    if mode == "Notes only":
        return "\n\n".join(rag.search(question)), "your notes", [], True
    if mode == "Web search":
        text, links = web.search(question)
        return text, "the web", links, True
    # Smart: use your notes if they are relevant, otherwise the AI's own knowledge
    chunks = rag.search(question)
    if chunks:
        return "\n\n".join(chunks), "your notes", [], True
    return "", "AI's own knowledge", [], False


def ask(display_text, instruction, context, source, links=None, use_context=True):
    """Send chat history + new request to the local model; save to memory."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]  # last 10 only
    ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append(
        {"role": "user", "content": build_prompt(instruction, context, use_context)}
    )
    try:
        reply = chat(messages)
    except Exception:
        st.error(
            "Couldn't reach the local model. Make sure Ollama is running "
            "and you ran: ollama pull llama3.2:3b"
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": display_text})
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "source": source, "links": links or []}
    )
    return reply


def run_task(name, topic):
    """Run a button task (summary, MCQs, ...) on the uploaded notes."""
    chunks = rag.search(topic, n=6) if topic else rag.sample(8)
    if not chunks:
        st.warning("Upload a PDF first (or try a different topic).")
        return
    label = f"{name}: {topic}" if topic else name
    with st.spinner("Working on it..."):
        ask(label, TASKS[name], "\n\n".join(chunks), "your notes")


# ---------- Sidebar ----------
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Answer from:",
    ["Smart (notes, then general)", "Notes only", "General knowledge", "Web search"],
    index=0,
    help="Smart = uses your PDF when it's relevant, otherwise answers from the "
         "AI's own knowledge (works with no PDF). Web search needs internet.",
)
speak_on = st.sidebar.checkbox("🔊 Read answers aloud")

st.sidebar.subheader("📄 Study material")
pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if pdf is not None and st.session_state.doc_name != pdf.name:
    with st.spinner("Reading and storing your notes..."):
        count = rag.add_pdf(pdf)
    if count == 0:
        st.sidebar.error("No readable text found in this PDF.")
    else:
        st.session_state.doc_name = pdf.name
        st.sidebar.success(f"Stored {count} chunks from {pdf.name}")

st.sidebar.subheader("🎧 Lecture audio")
audio_file = st.sidebar.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
if audio_file is not None and st.sidebar.button("Transcribe & summarize"):
    path = save_temp(audio_file.read(), os.path.splitext(audio_file.name)[1])
    with st.spinner("Transcribing (this can take a while)..."):
        transcript = voice.transcribe(path)
    os.remove(path)
    st.session_state.transcript = transcript
    with st.spinner("Summarizing..."):
        ask("🎧 Summarize my lecture audio", TASKS["Summary"],
            transcript[:6000], "audio transcript")

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------- Study tools ----------
st.subheader("Study tools")
topic = st.text_input("Topic (optional)", placeholder="e.g. BJT, transformer, shift register")
columns = st.columns(len(TASKS))
for column, name in zip(columns, TASKS):
    if column.button(name, use_container_width=True):
        run_task(name, topic)

if st.session_state.transcript:
    with st.expander("Show lecture transcript"):
        st.write(st.session_state.transcript)

# ---------- Voice question ----------
st.subheader("🎙️ Ask by voice")
recording = st.audio_input("Click the mic, speak, then stop")
voice_question = None
if recording is not None:
    data = recording.getvalue()
    key = hash(data)
    if st.session_state.last_voice != key:  # process each recording only once
        st.session_state.last_voice = key
        with st.spinner("Listening..."):
            try:
                voice_question = voice.transcribe_wav_bytes(data)
            except Exception as e:
                st.error(f"Couldn't read the recording: {e}")
        if voice_question:
            st.info(f"You said: {voice_question}")
        else:
            st.warning("I couldn't hear anything. Try again.")

# ---------- Chat history ----------
st.divider()
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("source"):
            st.caption(f"Source: {m['source']}")
            for link in m.get("links", []):
                st.caption(link)

# ---------- Chat input (typed or voice) ----------
typed = st.chat_input("Ask a question about your notes...")
question = typed or voice_question
if question:
    with st.chat_message("user"):
        st.markdown(question)

    context, source, links, use_context = get_context(question, mode)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask(question, question, context, source, links, use_context)
        st.markdown(reply)
        st.caption(f"Source: {source}")
        for link in links:
            st.caption(link)
        if speak_on:
            with st.spinner("Preparing voice..."):
                audio_bytes = voice.speak(reply)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav", autoplay=True)
            else:
                st.caption("(Voice output isn't available on this computer.)")