"""Text embeddings for Agent 3, using a local sentence-transformers model.

An embedding turns a text into a list of numbers (a vector) that captures
its meaning, so "developer" and "engineer" land close together even though
they are different words. The model runs on this computer: no API key, and
no resume text leaves the machine. It is downloaded once (about 90 MB) the
first time it is used, then loaded from the local cache.
"""
import os

import numpy as np

# Agent 3's embedding model. Change this one line to switch models.
EMBED_CONFIG = {"model": "sentence-transformers/all-MiniLM-L6-v2"}

# The model reads at most 256 tokens (about 180 words) and silently drops
# the rest. A resume is longer than that, so it is split into chunks of
# this many words and each chunk is embedded on its own.
CHUNK_WORDS = 120

_model = None      # loaded once, then reused


class EmbeddingError(Exception):
    """Raised when embeddings can't be produced (library not installed,
    model can't be downloaded, ...). Agent 3 catches it and uses TF-IDF."""


def _load_model():
    # The only function that touches sentence-transformers.
    # Tests replace it with a fake, so tests never download a model.
    global _model
    if _model is None:
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer   # slow import, so done here
            try:
                # Already downloaded: load from disk without going online
                _model = SentenceTransformer(EMBED_CONFIG["model"], local_files_only=True)
            except Exception:
                # First use on this computer: download it
                _model = SentenceTransformer(EMBED_CONFIG["model"])
        except Exception as e:
            raise EmbeddingError(f"could not load {EMBED_CONFIG['model']}: {type(e).__name__}") from None
    return _model


def embed(texts):
    """Returns one unit-length vector per text (a 2-D numpy array)."""
    model = _load_model()
    try:
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    except Exception as e:
        raise EmbeddingError(f"embedding failed: {type(e).__name__}") from None


def split_chunks(text, size=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def resume_job_similarities(resume_text, descriptions):
    """Cosine similarity between a resume and each job description.

    The resume is split into chunks. For each job, the similarity is that
    of the BEST-matching chunk: a job fits if some part of the resume (a
    project, a past job, the skills list) is about the same thing.
    Vectors have length 1, so the dot product IS the cosine similarity.
    """
    if not descriptions:
        return []
    chunk_vectors = embed(split_chunks(resume_text))
    job_vectors = embed(list(descriptions))
    similarities = chunk_vectors @ job_vectors.T          # chunks x jobs
    return [float(s) for s in similarities.max(axis=0)]
