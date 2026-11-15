# Equivalence gate for the batch indexing comparison: no timing number counts until these pass.
import os
import sys

import numpy as np
import pytest
from pyspark.sql import SparkSession

from pipeline.batch_index import run_single_process, run_spark
from pipeline.ivf_index import IVFIndex

DOCS_PATH = "data/mini_docs.tsv"
LIMIT = 200
MASTER = "local[4]"
ATOL = 1e-5

# Module level, not a fixture: both runs are slow, so do them once. The session
# is left open and dies with the pytest process; `master` can't change on a live
# SparkSession anyway, so there is nothing to reconfigure between these tests.
# workers must run the same interpreter as the driver, or they miss the conda
# env's torch and sentence-transformers
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

spark = SparkSession.builder.master(MASTER).appName("test_batch_index").getOrCreate()

single_ids, single_vectors = run_single_process(DOCS_PATH, limit=LIMIT)
spark_ids, spark_vectors = run_spark(spark, DOCS_PATH, limit=LIMIT)

def test_equivalence():
    assert set(single_ids) == set(spark_ids)
    # catches a duplicated row surviving on one side only, which equal sets hide
    assert len(single_ids) == len(spark_ids)
    assert len(set(spark_ids)) == len(spark_ids), "chunk_ids must be unique"

    # Spark's repartition shuffles, so row order differs: align by chunk_id before comparing.
    order = sorted(single_ids)
    single_by_id = dict(zip(single_ids, single_vectors))
    spark_by_id = dict(zip(spark_ids, spark_vectors))
    single_aligned = np.vstack([single_by_id[cid] for cid in order])
    spark_aligned = np.vstack([spark_by_id[cid] for cid in order])

    assert np.allclose(single_aligned, spark_aligned, atol=ATOL)

def test_ivf_handoff():
    index = IVFIndex()
    index.build(spark_vectors, spark_ids)

    # a chunk's own vector must retrieve itself: cosine 1.0, and nprobe always
    # covers the cluster its own centroid assigned it to
    probe = len(spark_ids) // 2
    results = index.search(spark_vectors[probe], k=5)

    top_chunk_id, top_score = results[0][0]
    assert top_chunk_id == spark_ids[probe]
    assert top_score == pytest.approx(1.0, abs=1e-3)
