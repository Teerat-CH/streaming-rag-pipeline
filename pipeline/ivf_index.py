import json
import os
import sys
from typing import List, Tuple
import numpy as np

from pipeline.chunk_embed import EMBED_DIM

import faiss

if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)

# number of clusters for k-means. should be around sqrt(N) where N is the number of vectors (chunks)
# faiss also wants ~39 training vectors per cluster, so N should be at least ~39 * NLIST
# TODO adjust this later when we try with the full dataset
NLIST = 32

INDEX_DIR = "index"
INDEX_FILE = "index.faiss"
META_FILE = "meta.json"

class IVFIndex:
    def __init__(self, nlist: int = NLIST):
        self.nlist = nlist
        self.index = None
        # map between the int ID faiss use to the chunk ID
        self.chunk_ids: List[str] = []
        self._seen = set()

    # make sure that a vectors are of the right format (# chunks x embed dimension)
    def _check_vectors(self, vectors: np.ndarray, chunk_ids: List[str]) -> np.ndarray:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
            raise ValueError(f"vectors must have shape (n, {EMBED_DIM}), got {vectors.shape}")
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(f"got {vectors.shape[0]} vectors but {len(chunk_ids)} chunk_ids")

        return np.ascontiguousarray(vectors, dtype=np.float32)

    # add new vectors with newly assigned numbers
    def _add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        start = len(self.chunk_ids)
        ids = np.arange(start, start + len(chunk_ids), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        self.chunk_ids.extend(chunk_ids)
        self._seen.update(chunk_ids)

    # build index from initial collection
    def build(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        vectors = self._check_vectors(vectors, chunk_ids)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids must be unique")
        if vectors.shape[0] < self.nlist:
            raise ValueError(f"need at least nlist={self.nlist} vectors to train, got {vectors.shape[0]}")

        quantizer = faiss.IndexFlatIP(EMBED_DIM)
        self.index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(vectors)

        self.chunk_ids = []
        self._seen = set()
        self._add(vectors, chunk_ids)

    # add vectors to existing index
    def add_batch(self, vectors: np.ndarray, chunk_ids: List[str]) -> int:
        if self.index is None:
            raise RuntimeError("call build() or load() before add_batch()")
        vectors = self._check_vectors(vectors, chunk_ids)

        keep = []
        new_ids = []
        batch_seen = set()
        for i, cid in enumerate(chunk_ids):
            if cid not in self._seen and cid not in batch_seen:
                keep.append(i)
                new_ids.append(cid)
                batch_seen.add(cid)

        if keep:
            self._add(vectors[keep], new_ids)
        return len(keep)

    def search(self, query_vecs: np.ndarray, k: int = 5, nprobe: int = 8) -> List[List[Tuple[str, float]]]:
        if self.index is None:
            raise RuntimeError("call build() or load() before search()")
        query_vecs = np.asarray(query_vecs, dtype=np.float32)
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        if query_vecs.shape[1] != EMBED_DIM:
            raise ValueError(f"query_vecs must have shape (q, {EMBED_DIM}), got {query_vecs.shape}")

        self.index.nprobe = min(nprobe, self.nlist)
        scores, ids = self.index.search(np.ascontiguousarray(query_vecs), k)

        results = []
        for row_scores, row_ids in zip(scores, ids):
            results.append([(self.chunk_ids[i], float(s)) for s, i in zip(row_scores, row_ids) if i != -1])
        return results

    # save index to disk
    def save(self, path: str = INDEX_DIR) -> None:
        if self.index is None:
            raise RuntimeError("nothing to save, call build() first")
        os.makedirs(path, exist_ok=True)

        # write to a temp file then rename, so a crash mid-save never leaves a half-written index
        index_path = os.path.join(path, INDEX_FILE)
        faiss.write_index(self.index, index_path + ".tmp")
        os.replace(index_path + ".tmp", index_path)

        meta_path = os.path.join(path, META_FILE)
        with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"nlist": self.nlist, "chunk_ids": self.chunk_ids}, f)
        os.replace(meta_path + ".tmp", meta_path)

    # load index from disk
    @classmethod
    def load(cls, path: str = INDEX_DIR) -> "IVFIndex":
        with open(os.path.join(path, META_FILE), encoding="utf-8") as f:
            meta = json.load(f)

        ivf = cls(nlist=meta["nlist"])
        ivf.index = faiss.read_index(os.path.join(path, INDEX_FILE))
        ivf.chunk_ids = meta["chunk_ids"]
        ivf._seen = set(ivf.chunk_ids)

        if ivf.index.ntotal != len(ivf.chunk_ids):
            raise ValueError(f"index has {ivf.index.ntotal} vectors but meta has {len(ivf.chunk_ids)} chunk_ids")
        return ivf
