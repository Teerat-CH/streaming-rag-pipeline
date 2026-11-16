# Batch indexing: single-process vs PySpark

This part reads documents, splits them into chunks with `chunk()`, turns the
chunks into vectors with `embed()`, and does that work two ways: in one plain
Python process, and with PySpark. Both ways must produce the same chunks and
the same vectors. Only then do we compare how long each one takes.

## Setup

Needs Python 3.12 and Java 17, the same JDK the streaming job uses.

```bash
# point JAVA_HOME at your Java 17 install, e.g. for Homebrew:
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
pip install -r requirements.txt
pip install matplotlib   # only needed for the benchmark plot
```

Run every command below from the repo root, so `from pipeline...` imports work.
Use `python`, not `python3`: on some machines `python3` points at a different
Python, and Spark fails when its workers run a different version than the
driver.

## Files

| File | What it does |
|---|---|
| `pipeline/batch_index.py` | `run_single_process()` and `run_spark()`: the two ways of doing chunk + embed, plus a quick demo |
| `experiment/test_batch_index.py` | Checks that both ways give the same output, and that the output builds an IVF index |
| `experiment/benchmark_batch.py` | Times one setting (mode + core count) and appends one row to a CSV |
| `experiment/run_batch_sweep.sh` | Runs the benchmark for every setting |
| `experiment/plot_batch.py` | Checks all rows still give the same output, then draws the timing plot |

## 1. Quick demo

```bash
python -m pipeline.batch_index
```

Runs the first 100 docs of `data/mini_docs.tsv` once in a single process and
once on Spark with 4 cores (`local[4]`). It prints:

- `single chunks: N seconds: S` and `spark chunks: N seconds: S`: how many
  chunks each way made, and how long chunk + embed took.
- `spark startup seconds: S`: how long Spark took to start. This is timed on
  its own because it is a fixed cost, not part of the work.
- `chunk_ids match`: both ways made exactly the same chunks.

This demo only shows that the code works. It is not a fair speed comparison:
the single-process run lets PyTorch use all cores, while Spark gives each of
its 4 workers one thread. Use the benchmark (section 3) for real numbers.

## 2. Tests

```bash
python -m pytest experiment/test_batch_index.py -v
```

Runs both ways on the first 200 docs. Takes about a minute.

- `test_equivalence`: both ways produce the same chunk_ids, the same number of
  them, and vectors equal within `1e-5` after lining them up by chunk_id.
- `test_ivf_handoff`: builds an `IVFIndex` from the Spark output and searches
  with one chunk's own vector. That chunk must come back first with a score of
  about 1.0.

`2 passed` means the batch output is correct and plugs into the index. A
`FutureWarning` about pandas 3 comes from PySpark itself and is harmless: this
code doesn't use Spark's pandas features.

## 3. Benchmark

### Get the 10k documents

The 10k-doc file lives on the `stream` branch. Copy it next to the repo (not
inside it, so git never sees it):

```bash
git fetch
git show origin/stream:data/mini_docs.tsv > ../docs_10k.tsv
```

### Run one setting

```bash
python -m experiment.benchmark_batch --docs ../docs_10k.tsv --mode spark --k 4 --limit 2000
```

- `--mode single` or `spark`: which way to run.
- `--k`: how many cores. Single mode tells PyTorch to use k threads; Spark mode
  runs `local[k]` with k workers of one thread each. Same k, same cores.
- `--limit`: how many docs, always the first N of the file.
- `--out`: which CSV to append to (default `experiment/results/batch_timings.csv`).
  Use a scratch path for test runs so they don't mix with real results.

It prints the row it wrote. If `--limit` is bigger than the file, it stops
with `--limit N exceeds M docs in PATH`.

### Run every setting

```bash
bash experiment/run_batch_sweep.sh ../docs_10k.tsv 2000
```

Runs single then Spark for k = 1, 2, 4 and 8: 8 rows in total. Each setting
runs as a fresh process, so nothing left over from one run speeds up or slows
down the next. Keep the laptop plugged in and leave it alone while it runs,
since other apps compete for the same cores. The k=1 runs are the slowest.

### Plot

```bash
python -m experiment.plot_batch
```

First it checks every row against the most recent single-process k=1 row for
the same docs file and limit: same chunk_ids (by hash), same vector shape,
vectors within `1e-5`. Then it plots run time against k, one line per mode,
one panel per docs file and limit, saved to `experiment/results/batch_scaling.png`.

- `gate passed for N dataset(s); wrote PATH`: all rows agree, the plot is saved.
- `equivalence gate failed, not plotting:` followed by the bad rows: some run
  produced different output. No plot is written. Find out why before trusting
  any number.
- `no single k=1 row for docs=... limit=...`: every docs file and limit needs a
  single-process k=1 run to compare against. Run the full sweep for it.

If a setting was run more than once, every copy is checked but only the most
recent one is plotted.

### What the CSV columns mean

| Column | Meaning |
|---|---|
| `timestamp` | When the run started |
| `docs`, `limit` | Which file and how many docs |
| `mode`, `k` | Single or Spark, and how many cores |
| `chunks` | How many chunks were made |
| `startup_s` | Spark session start time (0 for single) |
| `warmup_s` | Model loading before the timed run. Single: one `embed()` call. Spark: a small job on 4×k docs meant to load the model in every worker; Spark can still start fresh workers for the timed run, which reload it there |
| `run_s` | The timed chunk + embed on all `limit` docs. For Spark this also includes reading the file, shuffling and collecting results to the driver |
| `index_s` | Building the IVF index from the vectors. Always runs on one core on the driver, the same in both modes. At 2,000 docs it takes about 0.02 s, so it doesn't limit the speedup |
| `docs_per_s` | `limit / run_s` |
| `chunk_ids_sha256` | Fingerprint of the sorted chunk_ids, used by the plot check |
| `vectors_path` | The saved vectors for this run, used by the plot check |

### Where things are saved

- `experiment/results/`: the CSV and the plot. These are committed.
- `experiment/data/`: one `.npy` vectors file per run. These are ignored by
  `experiment/.gitignore` and can be deleted once the plot is final. At 10k
  docs they add up to several hundred MB.

### Results so far

2,000 docs (18,326 chunks) from the stream branch's 10k file, on a 12-thread
MacBook Pro, one run per setting. `run_s` in seconds:

| k | single | Spark | Spark vs single |
|---|---|---|---|
| 1 | 682.7 | 680.1 | 1.00× |
| 2 | 408.9 | 368.7 | 1.11× |
| 4 | 272.9 | 230.0 | 1.19× |
| 8 | 207.0 | 137.3 | 1.51× |

At one core both take the same time, so Spark's own overhead is negligible
here. With more cores Spark pulls ahead: from 1 to 8 cores it speeds up 5.0×,
the single process 3.3×. Likely reason: Spark runs k model copies on different
documents, while the single process splits each small matrix calculation
across threads, which scales worse for a model this size. Spark's fixed costs
(about 3 s startup, 5–10 s worker warm-up) are small next to this, and the IVF
build takes about 0.02 s, so it doesn't limit the speedup.

## Notes

The Spark side reads with `sc.textFile` and splits on tab rather than using
`spark.read.csv`. `data/data.py` writes unquoted, tab-joined fields with tabs
and newlines stripped. A plain split parses that exactly like the baseline's
`csv.QUOTE_NONE` reader; `spark.read.csv` handles quotes differently.

The benchmark sets `TOKENIZERS_PARALLELISM=false`. Otherwise the tokenizer
starts its own threads, and a "k=1" run would quietly use more than one core.

A full 10k sweep takes several hours, because the single k=1 reference run
alone is slow. Run it overnight.

Spark splits the docs into k partitions. Doc lengths range from 1 to about
183,000 characters, so one partition can end up with most of the long docs and
finish last while other cores sit idle. This is likely one reason Spark's
8-core speedup is 5.0× rather than 8×.