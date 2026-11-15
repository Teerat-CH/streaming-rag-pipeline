# Batch indexing: the single-process baseline and the PySpark path for the
# comparison. The two paths share no helpers on purpose, so a reader can check line-by-line that they do identical work.
import csv
import os
import sys
import time

import numpy as np

from pipeline.chunk_embed import EMBED_DIM, chunk, embed

DOCS_PATH = "data/mini_docs.tsv"
DEMO_LIMIT = 100
DEMO_MASTER = "local[4]"
FIELD_SIZE_LIMIT = 10 * 1024 * 1024
EXPECTED_HEADER = ["doc_id", "url", "title", "body"]

def run_single_process(path, limit=None):
    csv.field_size_limit(FIELD_SIZE_LIMIT)

    chunk_ids = []
    texts = []
    doc_count = 0
    with open(path, encoding="utf-8", newline="") as f:
        # data/data.py writes raw tab-joined fields with tabs and newlines
        # stripped and no quoting, so QUOTE_NONE: a body starting with a quote is literal text, not an opening delimiter.
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

def _index_partition(lines):
    # One torch thread per worker: N workers x M threads oversubscribe the CPU
    # and corrupt the measurement. Set before torch is imported by chunk_embed.
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)

    from pipeline.chunk_embed import chunk, embed

    chunk_ids = []
    texts = []
    for line in lines:
        doc_id, _url, title, body = line.split("\t")
        for chunk_id, _doc_id, text in chunk(doc_id, title, body):
            chunk_ids.append(chunk_id)
            texts.append(text)

    # one embed() call per partition, never per row: a row-at-a-time UDF woulddefeat embed()'s internal batching 
    # and make Spark look slow for the wrong reason. The model loads once per executor process and is then reused.
    vectors = embed(texts)
    return zip(chunk_ids, vectors)

def run_spark(spark, path, limit=None, num_partitions=None):
    sc = spark.sparkContext
    lines = sc.textFile(path)

    # zipWithIndex gives line order, so index 0 is the header. 
    # # indices 1..limit are exactly the first `limit` rows the baseline reads.
    indexed = lines.zipWithIndex()
    header = indexed.filter(lambda pair: pair[1] == 0).map(lambda pair: pair[0]).first()
    assert header.split("\t") == EXPECTED_HEADER, header
    rows = indexed.filter(
        lambda pair: pair[1] >= 1 and (limit is None or pair[1] <= limit)
    ).map(lambda pair: pair[0])

    # A small TSV reads as 1-2 partitions and leaves most cores idle, so repartition explicitly to at least the core count.
    if num_partitions is None:
        num_partitions = sc.defaultParallelism
    rows = rows.repartition(num_partitions)

    pairs = rows.mapPartitions(_index_partition).collect()
    if not pairs:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)

    chunk_ids = [chunk_id for chunk_id, _vector in pairs]
    vectors = np.vstack([vector for _chunk_id, vector in pairs])
    return chunk_ids, vectors

if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # workers must run the same interpreter as the driver, or they miss the
    # conda env's torch and sentence-transformers
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    t0 = time.perf_counter()
    single_ids, single_vectors = run_single_process(DOCS_PATH, limit=DEMO_LIMIT)
    single_seconds = time.perf_counter() - t0

    # Timed separately: session startup is a fixed cost, not part of chunk+embed.
    t0 = time.perf_counter()
    spark = SparkSession.builder.master(DEMO_MASTER).appName("batch_index").getOrCreate()
    startup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    spark_ids, spark_vectors = run_spark(spark, DOCS_PATH, limit=DEMO_LIMIT)
    spark_seconds = time.perf_counter() - t0
    spark.stop()

    print(f"single  chunks: {len(single_ids)}  seconds: {single_seconds:.2f}")
    print(f"spark   chunks: {len(spark_ids)}  seconds: {spark_seconds:.2f}")
    print(f"spark startup seconds: {startup_seconds:.2f}")

    assert len(set(spark_ids)) == len(spark_ids), "chunk_ids must be unique"
    assert set(single_ids) == set(spark_ids), (
        f"chunk_id sets differ: "
        f"{len(set(single_ids) - set(spark_ids))} only single, "
        f"{len(set(spark_ids) - set(single_ids))} only spark"
    )
    print("chunk_ids match")
