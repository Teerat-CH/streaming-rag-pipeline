# chunk + embed for the indexing pipeline
# called per document from the pyspark batch / streaming jobs, so no file I/O here

from typing import List, Tuple

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# size of chunk
CHUNK_SIZE = 1000
# overlap
CHUNK_OVERLAP = 200
# cap on title length in the prefix
MAX_TITLE_CHARS = 200

# embedding model to use
MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-v3"
EMBED_DIM = 384
# texts per batch when encoding
BATCH_SIZE = 64
# only cpu for now
DEVICE = "cpu"


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
    )


def chunk(doc_id: str, title: str, body: str) -> List[Tuple[str, str, str]]:
    # spearate by default: "\n\n", "\n", " "
    pieces = splitter.split_text(body)

    # prefix the title to each chunk if title is not empty
    title = title.strip()[:MAX_TITLE_CHARS].strip()
    prefix = f"{title}. " if title else ""

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append((f"{doc_id}_{i}", doc_id, prefix + piece))
    return chunks


# load the model once and reuse it for all calls to embed()
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    # return (0, 384) float32 if empty
    if len(texts) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model = _get_model()
    
    # Encode the texts and return the embeddings (n, 384) float32
    return model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        normalize_embeddings=True,
        show_progress_bar=False
        )
