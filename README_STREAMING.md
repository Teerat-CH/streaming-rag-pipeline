# Streaming chunk embed pipeline
```text
terminal line usage run from project root directory
example: python -m data.data
used Python 3.12 and Java 17
if MS MARCO dataset not available locally yet, run command above before using venv
then continue with data.prepare_stream, pipeline.stream_flink, and data. prepare_stream in that order
```

```text
data/data.py             # run with python -m data.data, sets up MS MARCO dataset, current using 10,000 documents
data/prepare_stream.py   # run with python -m data.prepare_stream, deterministic static/stream split + arrival schedule
data/release_stream.py   # run with python -m data.release_stream, timed document out of chunked embeddings
pipeline/chunk_embed.py  # shared chunk() and embed(), don't use .py directly, is called in stream_flink.py
pipeline/stream_flink.py # run with python -m pipeline.stream_flink, PyFlink streaming of data
data.clear_inbox.py      # run with python -m data.clear_inbox, used for cleaning up old data before starting new streams, testing use
```

```text
storage and time estimates based off Steven's computer:
streaming 8000 documents and outputting their embeds took 15~ minutes, with a folder size of 300~ MB
full MS MARCO's 3.2 million documents would result in close to 120 GB at that rate
original MS MARCO dataset is 22 GB
```
