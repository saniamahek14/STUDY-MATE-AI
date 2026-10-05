# core/prompts.py
# Every prompt used by StudyMate lives here.

SYSTEM_PROMPT = """You are StudyMate, a friendly study tutor and general knowledge assistant.
Rules:
- Explain in simple words, step by step.
- Use short paragraphs, bullet points, and small tables where helpful.
- You can answer any question, not only study topics.
- Never invent facts, formulas, or numbers. If you are unsure, say so."""

# Each button = the same RAG pipeline + a different instruction.
TASKS = {
    "Summary": "Summarize this material in 8 clear bullet points.",
    "Key points": "List the most important points and formulas to remember for an exam. Use bullet points.",
    "MCQs": (
        "Create 5 multiple-choice questions from this material. "
        "Give 4 options (A, B, C, D) for each and mark the correct answer "
        "at the end of each question."
    ),
    "Flashcards": (
        "Create 8 flashcards from this material. "
        "Use this exact format:\nQ: question\nA: short answer"
    ),
    "Simple": "Explain this material like I'm a beginner. Use simple words and one real-life example.",
}


def build_prompt(instruction, context, kind):
    """
    kind = "notes": answer from the user's PDF notes
    kind = "web":   answer using web results (plus own knowledge)
    kind = "none":  answer from the AI's own knowledge
    """
    if kind == "notes":
        return (
            f"Context from the user's notes:\n{context}\n\n"
            f"Using ONLY this context, do the following. If the context does not "
            f"contain the answer, say: \"I couldn't find this in your notes.\"\n"
            f"{instruction}"
        )
    if kind == "web":
        return (
            f"Web search results:\n{context}\n\n"
            f"Answer the request below. Use the web results where they help and "
            f"add your own knowledge where needed.\n{instruction}"
        )
    return instruction