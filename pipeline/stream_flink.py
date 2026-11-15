# create stream environment in seperate terminal using python -m data.stream_flink
# PyFlink streaming: JSONL files -> shared chunk() + embed() -> JSON output

import json
from pathlib import Path

from pyflink.common import Encoder

ROOT = Path(__file__).resolve().parents[1]

# settings for the flink streaming job
INBOX = ROOT / "artifacts" / "inbox"
OUTPUT_DIR = ROOT / "artifacts" / "embeddings"
POLL_SECONDS = 1
EMBED_BATCH_SIZE = 64
PARALLELISM = 1
CHECKPOINT_MS = 10000

def process_document(line: str, embed_batch_size: int = 64):
    """Shared pure-Python transformation used by the PyFlink flat_map worker."""
    from pipeline.chunk_embed import chunk, embed

    if embed_batch_size < 1:
        raise ValueError("embed_batch_size must be positive")
    
    # read one incoming document
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError("expected a JSON object for each document")
    for name in ("doc_id", "title", "body"):
        if name not in record or record[name] is None:
            raise ValueError(f"Stream document missing {name}")
        
    # split the document into chunks
    pieces = chunk(str(record["doc_id"]), record["title"] or "", record["body"] or "")

    # embed chunks in batches instead of all at once
    for start in range(0, len(pieces), embed_batch_size):
        subset = pieces[start:start + embed_batch_size]
        vectors = embed([item[2] for item in subset])

        # include the chunk id, source id, and arrival metadata
        for (chunk_id, doc_id, text), vector in zip(subset, vectors):
            yield json.dumps({"chunk_id": chunk_id, "doc_id": doc_id, "text": text,
                              "arrival_time": record.get("arrival_time"),
                              "released_at": record.get("released_at"),
                              "embedding": vector.tolist()}, ensure_ascii=False)

class ChunkAndEmbed:
    """PyFlink flat_map callable, this class lets workers reuse imported model state."""
    def __init__(self, embed_batch_size: int):
        self.embed_batch_size = embed_batch_size

    def __call__(self, line):
        yield from process_document(line, self.embed_batch_size)

def main():
    if min(POLL_SECONDS, EMBED_BATCH_SIZE, PARALLELISM, CHECKPOINT_MS) < 1:
        raise ValueError("numeric streaming settings must be positive")

    from pyflink.common import Duration, WatermarkStrategy
    from pyflink.common.typeinfo import Types
    from pyflink.datastream import RuntimeExecutionMode, StreamExecutionEnvironment
    from pyflink.datastream.connectors.file_system import (
        FileSource, StreamFormat, FileSink, RollingPolicy, OutputFileConfig
    )

    # set up the flink streaming environment
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    env.set_parallelism(PARALLELISM)
    env.enable_checkpointing(CHECKPOINT_MS)

    # keep checking the folder for newly released documents
    source = (FileSource.for_record_stream_format(
        StreamFormat.text_line_format(), str(INBOX.resolve()))
        .monitor_continuously(Duration.of_seconds(POLL_SECONDS))
        .build())
    
    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "jsonl-document-files")

    # process incoming documents and print the resulting vectors
    chunks = stream.flat_map(ChunkAndEmbed(EMBED_BATCH_SIZE), output_type=Types.STRING())

    # save chunks and embeddings to JSONL
    sink = (
        FileSink.for_row_format(
            str(OUTPUT_DIR.resolve()),
            Encoder.simple_string_encoder("UTF-8")
        )
        .with_output_file_config(
            OutputFileConfig.builder()
            .with_part_prefix("embeddings")
            .with_part_suffix(".jsonl")
            .build()
        )
        .with_rolling_policy(
            RollingPolicy.on_checkpoint_rolling_policy()
        )
            .build()
        )

    chunks.sink_to(sink)

    print(f"Monitoring {INBOX.resolve()} (Ctrl+C to stop)", flush=True)
    print(f"Writing embeddings to {OUTPUT_DIR.resolve()}", flush=True)
    env.execute("RAG PyFlink Chunk Embed Stream")

if __name__ == "__main__":
    main()