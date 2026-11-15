# One invocation = one (mode, k, limit) setting = one appended CSV row.
# Run each setting in its own process (see run_batch_sweep.sh) so no Spark or
# torch state carries between settings.
import os
import sys

# Must be set before torch / tokenizers / pyspark are imported.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
# Workers must run the driver's Python (we hit 3.11 vs 3.12 without this).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

import argparse
import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_OUT = "experiment/results/batch_timings.csv"
VECTORS_DIR = "experiment/data"
FIELD_SIZE_LIMIT = 10 * 1024 * 1024
WARMUP_DOCS_PER_CORE = 4
COLUMNS = [
    "timestamp", "docs", "mode", "k", "limit", "chunks", "startup_s",
    "warmup_s", "run_s", "index_s", "docs_per_s", "chunk_ids_sha256",
    "vectors_path",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--mode", required=True, choices=["single", "spark"])
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.k < 1 or args.limit < 1:
        parser.error("--k and --limit must be >= 1")
    return args

def count_data_rows(path):
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return sum(1 for _ in reader) - 1  # minus header

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0

def run_single(docs, k, limit):
    import torch

    from pipeline.batch_index import run_single_process
    from pipeline.chunk_embed import embed

    torch.set_num_threads(k)
    _, warmup_s = timed(embed, ["warm up"])  # model load + first call
    (chunk_ids, vectors), run_s = timed(run_single_process, docs, limit)
    return chunk_ids, vectors, 0.0, warmup_s, run_s

def run_spark_mode(docs, k, limit):
    from pyspark.sql import SparkSession

    from pipeline.batch_index import run_spark

    builder = SparkSession.builder.master(f"local[{k}]").appName("benchmark_batch")
    spark, startup_s = timed(builder.getOrCreate)
    try:
        # Small run with k partitions so each worker loads the model before the
        # timed run. Spark may still start fresh workers for the timed job, which
        # then reload the model inside run_s (a few seconds, small at 2,000 docs).
        _, warmup_s = timed(
            run_spark, spark, docs, limit=WARMUP_DOCS_PER_CORE * k, num_partitions=k
        )
        (chunk_ids, vectors), run_s = timed(
            run_spark, spark, docs, limit, num_partitions=k
        )
    finally:
        spark.stop()
    return chunk_ids, vectors, startup_s, warmup_s, run_s

def save_sorted_vectors(chunk_ids, vectors, docs, mode, k, limit, stamp):
    # Sorted by chunk_id so plot_batch.py can compare runs row-for-row.
    order = sorted(range(len(chunk_ids)), key=chunk_ids.__getitem__)
    out_dir = Path(VECTORS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Timestamp in the name: reruns must not overwrite an older row's vectors.
    path = out_dir / f"vectors_{Path(docs).stem}_{mode}_k{k}_n{limit}_{stamp}.npy"
    np.save(path, vectors[order])
    return str(path)

def append_row(out, row):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def main():
    args = parse_args()
    available = count_data_rows(args.docs)
    if args.limit > available:
        # Otherwise docs_per_s = limit / run_s would overstate throughput.
        sys.exit(f"--limit {args.limit} exceeds {available} docs in {args.docs}")

    now = datetime.now()
    runner = run_single if args.mode == "single" else run_spark_mode
    chunk_ids, vectors, startup_s, warmup_s, run_s = runner(args.docs, args.k, args.limit)

    from pipeline.ivf_index import IVFIndex

    # Serial and identical in both modes; timed apart so it can't hide the speedup.
    _, index_s = timed(IVFIndex().build, vectors, chunk_ids)

    vectors_path = save_sorted_vectors(
        chunk_ids, vectors, args.docs, args.mode, args.k, args.limit,
        now.strftime("%Y%m%dT%H%M%S"),
    )
    sha = hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()
    row = {
        "timestamp": now.isoformat(timespec="seconds"),
        "docs": args.docs,
        "mode": args.mode,
        "k": args.k,
        "limit": args.limit,
        "chunks": len(chunk_ids),
        "startup_s": f"{startup_s:.4f}",
        "warmup_s": f"{warmup_s:.4f}",
        "run_s": f"{run_s:.4f}",
        "index_s": f"{index_s:.4f}",
        "docs_per_s": f"{args.limit / run_s:.4f}",
        "chunk_ids_sha256": sha,
        "vectors_path": vectors_path,
    }
    append_row(args.out, row)
    print(row)

if __name__ == "__main__":
    main()
