# IVF vector index for the indexing pipeline (using faiss IndexIVFFlat)

from typing import List, Tuple

import faiss
import numpy as np

from pipeline.chunk_embed import EMBED_DIM

# number of clusters for k-means. should be around sqrt(N) where N is the number of documents
NLIST = 32

class IVFIndex:
    def __init__(self, nlist: int = NLIST):
        self.nlist = nlist
        self.index = None
        # faiss int id -> chunk_id, the int id is the position in this list
        self.chunk_ids: List[str] = []

    def _check_vectors(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        pass

    def build(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        # train k-means centroids on vectors then add them
        # vectors: (n, 384) float32 normalized, row i belongs to chunk_ids[i]
        pass

    def add_batch(self, vectors: np.ndarray, chunk_ids: List[str]) -> int:
        # add new vectors to the existing clusters, no retraining and return the number of vectors actually added
        pass

    def search(self, query_vecs: np.ndarray, k: int = 5, nprobe: int = 8) -> List[List[Tuple[str, float]]]:
        # query_vecs: (q, 384) float32 normalized, from embed()
        # return for each query a list of (chunk_id, score), best first
        pass

    def save(self, path: str) -> None:
        # write the faiss index and the chunk_id map to the path
        pass

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        # read back what save() wrote, returns a new IVFIndex
        pass
