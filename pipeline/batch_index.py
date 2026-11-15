# Single-process baseline for the batch indexing comparison.
# Deliberately kept free of shared helpers with the (future) Spark path so a
# reader can check line-by-line that both paths do identical work.

import csv
import time

from pipeline.chunk_embed import chunk, embed

DOCS_PATH = "data/mini_docs.tsv"
DEMO_LIMIT = 100
# bodies run to hundreds of KB, well past csv's default field cap
FIELD_SIZE_LIMIT = 10 * 1024 * 1024
EXPECTED_HEADER = ["doc_id", "url", "title", "body"]


def run_single_process(path, limit=None):
    csv.field_size_limit(FIELD_SIZE_LIMIT)

    chunk_ids = []
    texts = []
    doc_count = 0
    with open(path, encoding="utf-8", newline="") as f:
        # data/data.py writes raw tab-joined fields with tabs and newlines
        # stripped and no quoting, so QUOTE_NONE: a body starting with a quote
        # is literal text, not an opening delimiter.
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        assert header == EXPECTED_HEADER, header
        for doc_id, _url, title, body in reader:
            if limit is not None and doc_count >= limit:
                break
            doc_count += 1
            for chunk_id, _doc_id, text in chunk(doc_id, title, body):
                chunk_ids.append(chunk_id)
                texts.append(text)

    # one call: embed() batches internally at 64, and per-text calls would
    # defeat that and make the baseline look slow for the wrong reason
    vectors = embed(texts)
    return chunk_ids, vectors


if __name__ == "__main__":
    t0 = time.perf_counter()
    chunk_ids, vectors = run_single_process(DOCS_PATH, limit=DEMO_LIMIT)
    elapsed = time.perf_counter() - t0

    # chunk_id is f"{doc_id}_{i}", so the prefix recovers the doc. Counts only
    # docs that produced chunks, which is what actually got embedded.
    doc_count = len({cid.rsplit("_", 1)[0] for cid in chunk_ids})

    assert len(set(chunk_ids)) == len(chunk_ids), "chunk_ids must be unique"
    assert vectors.shape[0] == len(chunk_ids)

    print(f"docs:    {doc_count}")
    print(f"chunks:  {len(chunk_ids)}")
    print(f"vectors: {vectors.shape} {vectors.dtype}")
    print(f"seconds: {elapsed:.2f}")
