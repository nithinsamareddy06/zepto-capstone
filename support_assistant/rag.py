"""
Module 3 - Support Assistant
Retrieval helper: embeds a query with all-MiniLM-L6-v2 and pulls the
top-k most similar chunks from the persisted ChromaDB collection built by
ingest.py. This step always runs for real in both MOCK_LLM=1 and
MOCK_LLM=0 modes - embedding and ChromaDB need no API key and no network
call once the model is cached locally.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL_NAME

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_top_k(query: str, k: int = 3) -> list[dict]:
    """Returns up to k chunks as [{"id":..., "text":..., "distance":...}, ...],
    ordered by cosine distance (ChromaDB's default space) ascending."""
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    for doc_id, text, dist in zip(ids, documents, distances):
        chunks.append({"id": doc_id, "text": text, "distance": dist})
    return chunks
