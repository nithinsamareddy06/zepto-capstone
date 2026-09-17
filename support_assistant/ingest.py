"""
Module 3 - Support Assistant
Ingestion + embedding stage.

Loads the 8 policy documents from docs/, chunks each one (one chunk per
document, since each is short and already topically coherent), embeds each
chunk locally with sentence-transformers/all-MiniLM-L6-v2 (no API key, no
network cost beyond the one-time model download), and stores the vectors
in a persistent ChromaDB collection at ./chroma_store.

Run: python ingest.py
"""

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    """One chunk per document file - each doc is short and single-topic,
    so document-level chunking preserves full context per chunk."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_index():
    chunks = load_documents()
    print(f"Loaded {len(chunks)} document chunks from {DOCS_DIR}/")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # start clean each run so re-ingestion doesn't duplicate rows
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"Indexed {len(ids)} chunks into ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_PATH}/")


if __name__ == "__main__":
    build_index()
