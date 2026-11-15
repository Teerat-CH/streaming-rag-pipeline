# Batch indexing: single-process vs PySpark

Needs Python 3.12 and Java 17, the same JDK the streaming job uses.

```bash
# point JAVA_HOME at your Java 17 install, e.g. for Homebrew:
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
pip install -r requirements.txt

# run both from the repo root, so `from pipeline...` imports resolve
python -m pipeline.batch_index
python -m pytest experiment/test_batch_index.py -v
```

## Notes

`python -m pipeline.batch_index` indexes the first 100 docs of
`data/mini_docs.tsv` twice, once single-process and once on a `local[4]`
SparkSession, and prints the chunk count and seconds for each, the Spark
startup seconds on their own line, and `chunk_ids match`. Session startup is
timed separately because it is a fixed cost, not part of chunk+embed.

The two tests run on the first 200 docs. `test_equivalence` checks that both
paths produce the same chunk_ids, the same number of them, and vectors equal
within `atol=1e-5` after aligning by chunk_id. `test_ivf_handoff` builds an
`IVFIndex` from the Spark output and checks that searching with one chunk's own
vector returns that chunk first.

The Spark side reads with `sc.textFile` and splits on tab rather than
`spark.read.csv`: `data/data.py` writes unquoted tab-joined fields with tabs and
newlines stripped, which a plain split parses exactly like the baseline's
`csv.QUOTE_NONE` reader, while `spark.read.csv`'s quote handling does not.

At 100 docs Spark is slower. Part of that is fixed cost (Spark startup and
four model loads), and part is that the single-process run is not single-core:
PyTorch already spreads it across all cores. A comparison at equal core counts
is still to come.