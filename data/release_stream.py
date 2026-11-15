# release stream documents (chunked embeddings) into folder
# run with python -m data.release_stream

import json
import os
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# settings for releasing streaming documents
MANIFEST = ROOT / "artifacts" / "prepared" / "stream_manifest.jsonl"
INBOX = ROOT / "artifacts" / "inbox"

# used to simulate arrival times for stream documents; can be set to 1.0 for real-time, or >1.0 for faster than real-time
# 10x faster than simulated arrival timestamps here
SPEED = 10.0

#-------------------------------------
# IMPORTANT TO CHANGE LATER!!!!!!!!!!!
# preview first 10 results for now;
# set to None for all stream documents
MAX_DOCS = None
#-------------------------------------

# True releases immediately, ignoring timestamps
NO_WAIT = False

def release(manifest: Path, inbox: Path, speed: float = 1.0,
            max_docs: int | None = None, no_wait: bool = False) -> int:
    # check the release speed
    if speed <= 0:
        raise ValueError("speed must be positive")
    inbox.mkdir(parents=True, exist_ok=True)

    # write temporary files outside the watched folder
    # used to avoid partial files being picked up by the stream
    staging = inbox.parent / f".{inbox.name}_staging"
    staging.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    count = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if max_docs is not None and count >= max_docs:
                break
            doc = json.loads(line)
            doc_id = str(doc["doc_id"])
            offset = float(doc["arrival_offset_seconds"])

            # wait until the scheduled arrival time
            if not no_wait:
                delay = t0 + offset / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            # give each released document a unique file name

            target = inbox / f"document_{count:08d}.jsonl"
            if target.exists():
                raise FileExistsError(f"{target} exists: use a fresh INBOX directory for each run")
            record = {k: doc[k] for k in ("doc_id", "url", "title", "body", "arrival_time")}
            record["released_at"] = time.time()
            tmp = staging / f"{uuid.uuid4().hex}.tmp"
            try:
                with tmp.open("w", encoding="utf-8", newline="\n") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                # move completed file into the watched folder
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            count += 1
            print(f"released {doc_id} -> {target.name}", flush=True)
    return count

def main():
    count = release(MANIFEST, INBOX, SPEED, MAX_DOCS, NO_WAIT)
    print(f"Released {count} documents")


if __name__ == "__main__":
    main()