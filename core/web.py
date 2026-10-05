# core/web.py
# OPTIONAL web search (DuckDuckGo, no API key). Needs internet.

from ddgs import DDGS


def search(query, n=4):
    """Returns (text_for_the_prompt, list_of_links)."""
    try:
        with DDGS(timeout=8) as ddgs:
            results = list(ddgs.text(query, max_results=n))
    except Exception:
        return "", []
    text = "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
    links = [r["href"] for r in results]
    return text, links