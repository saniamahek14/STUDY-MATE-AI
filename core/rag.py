# core/rag.py
# Local RAG: PDF -> chunks -> embeddings -> ChromaDB -> search.

import re
import unicodedata

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual model: works for English, French, Hindi, etc. (downloads once, ~470 MB)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="chroma_db")
COLLECTION = "notes"


def clean_text(text):
    """Remove characters that break the tokenizer (broken unicode, null bytes)
    and turn ligatures like 'ﬁ' into normal letters."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = text.replace("\x00", " ")
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t]+", " ", text)


def read_pdf(file):
    """Extract all text from a PDF (path or uploaded file)."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += clean_text(page.extract_text() or "") + "\n"
    return text


def chunk_text(text, size=500, overlap=100):
    """Cut long text into small overlapping pieces."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def add_pdf(file):
    """Read, chunk, embed and store a PDF. Returns the number of chunks."""
    chunks = chunk_text(read_pdf(file))
    if not chunks:
        return 0  # scanned PDF with no readable text

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    return len(chunks)


def _get_collection():
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return None


def search(question, n=4, max_distance=0.6):
    """Return chunks related to the question (lower distance = more similar)."""
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []
    q_embedding = embedder.encode([clean_text(question)]).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=min(n, collection.count()),
    )
    docs = result["documents"][0]
    distances = result["distances"][0]
    return [d for d, dist in zip(docs, distances) if dist <= max_distance]


def sample(n=8):
    """Pick n evenly spaced chunks (for whole-document summaries)."""
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []
    data = collection.get()
    pairs = sorted(zip(data["ids"], data["documents"]))
    docs = [d for _, d in pairs]
    step = max(1, len(docs) // n)
    return docs[::step][:n]