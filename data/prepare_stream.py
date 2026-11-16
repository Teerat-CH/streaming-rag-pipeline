# run with python -m data.prepare_stream
# this file helps prepare an initial static set of documents and a stream of documents with simulated arrival times
# can make it fully streamed later, but for now it's easier to prepare the static set first and then stream 
# the rest for testing the pipeline's components

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("doc_id", "url", "title", "body")

# settings for preparing the dataset
INPUT_PATH = ROOT / "data" / "mini_docs.tsv"
OUTPUT_DIR = ROOT / "artifacts" / "prepared"
# STREAM_FRACTION = 0.8 means with 10,000 documents, 8,000 will be streamed and 2,000 will be the initial static
STREAM_FRACTION = 0.8
SEED = 67
INTERVAL_SECONDS = 1.0
START_TIME = datetime.now(timezone.utc).isoformat()

def prepare(input_path: Path, output_dir: Path, stream_fraction: float = 0.25,
            seed: int = 67, interval_seconds: float = 1.0,
            start_time: str = None) -> dict:
    
    if start_time is None:
        start_time = datetime.now(timezone.utc).isoformat()
    
    # check the input settings
    if not 0 < stream_fraction < 1:
        raise ValueError("stream_fraction must be strictly between 0 and 1")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("start_time needs an explicit timezone, e.g. +00:00")
    
    # read the exported original MS MARCO documents from the tsv file
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. First run: python -m data.data"
        )
    # max size too large
    csv.field_size_limit(10 * 1024 * 1024)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Input TSV must include columns: {FIELDS}")
        docs = [{key: row[key] for key in FIELDS} for row in reader]

    # make sure each document has a unique id
    ids = [d["doc_id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("Input has duplicate document IDs")
    if len(docs) < 2:
        raise ValueError("Need at least two documents")

    # shuffle and split into static / streaming documents
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_stream = max(1, min(len(docs) - 1, round(len(docs) * stream_fraction)))
    streaming = docs[:n_stream]
    static = docs[n_stream:]

    # write the static documents to tsv
    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / "static_docs.tsv"
    with static_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(static)

    # write streaming documents with simulated arrival times
    manifest_path = output_dir / "stream_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        for i, doc in enumerate(streaming):
            record = dict(doc)
            record["arrival_offset_seconds"] = i * interval_seconds
            record["arrival_time"] = (start + timedelta(seconds=i * interval_seconds)).isoformat()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"static_docs": len(static), "stream_docs": len(streaming),
            "static_path": str(static_path), "manifest_path": str(manifest_path)}

def main():
    result = prepare(INPUT_PATH, OUTPUT_DIR, STREAM_FRACTION, SEED,
                     INTERVAL_SECONDS, START_TIME)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()