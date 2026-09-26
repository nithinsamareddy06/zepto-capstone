"""
Support Assistant - Ingestion & Embedding
================================================================
Loads the 8 policy documents in docs/, chunks them (one chunk per document -
each is already short and topically self-contained), embeds each chunk with
sentence-transformers' all-MiniLM-L6-v2, and stores them in a persistent
ChromaDB collection on disk (./chroma_db).

Run standalone: python3 ingest.py
Also imported by graph.py to get a ready-to-query collection.
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_documents():
    """One chunk per document; each doc is short and topically self-contained."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0 and not reset:
        return collection

    chunks = load_documents()
    model = get_embedding_model()
    embeddings = model.encode([c["text"] for c in chunks]).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
    )
    return collection


def query_collection(collection, query_text: str, top_k: int = 3):
    model = get_embedding_model()
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    matches = []
    for doc_id, doc_text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        matches.append({"id": doc_id, "text": doc_text, "distance": distance})
    return matches


if __name__ == "__main__":
    collection = build_collection(reset=True)
    print(f"Indexed {collection.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")

    # quick smoke test
    test_query = "How long do I have to return a damaged item?"
    hits = query_collection(collection, test_query, top_k=3)
    print(f"\nTop matches for: '{test_query}'")
    for h in hits:
        print(f"  [{h['id']}] (distance={h['distance']:.4f}) {h['text'][:100]}...")
