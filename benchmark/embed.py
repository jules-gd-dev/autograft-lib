"""Real embedding helper using sentence-transformers (all-MiniLM-L6-v2, CPU)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load the embedding model once (cached for the process lifetime)."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single string into a normalised float vector."""
    model = load_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=float).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many strings efficiently in one batched call."""
    model = load_model()
    mat = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(mat, dtype=float).tolist()
