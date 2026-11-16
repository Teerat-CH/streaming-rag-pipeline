# Equivalence gate + scaling plot for batch_timings.csv.
# Usage: python -m experiment.plot_batch [CSV] [PNG]
# Refuses to plot unless every row matches its (docs, limit) reference run:
# same chunk_ids (sha256) and vectors within ATOL after aligning by chunk_id.
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = "experiment/results/batch_timings.csv"
DEFAULT_PNG = "experiment/results/batch_scaling.png"
ATOL = 1e-5
REF_MODE = "single"
REF_K = 1

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def group_by_dataset(rows):
    # File order is kept, so the last row in each group is the most recent.
    groups = {}
    for row in rows:
        groups.setdefault((row["docs"], int(row["limit"])), []).append(row)
    return groups

def find_reference(key, rows):
    refs = [r for r in rows if r["mode"] == REF_MODE and int(r["k"]) == REF_K]
    if not refs:
        sys.exit(f"no {REF_MODE} k={REF_K} row for docs={key[0]} limit={key[1]}")
    return refs[-1]

def row_label(row):
    return f"{row['timestamp']} {row['mode']} k={row['k']}"

def check_group(key, rows):
    ref = find_reference(key, rows)
    ref_vectors = np.load(ref["vectors_path"])
    failures = []
    for row in rows:
        if row["chunk_ids_sha256"] != ref["chunk_ids_sha256"]:
            failures.append(f"{row_label(row)}: chunk_ids differ")
            continue
        vectors = np.load(row["vectors_path"])
        if vectors.shape != ref_vectors.shape:
            failures.append(f"{row_label(row)}: shape {vectors.shape} != {ref_vectors.shape}")
        elif not np.allclose(vectors, ref_vectors, atol=ATOL):
            diff = np.abs(vectors - ref_vectors).max()
            failures.append(f"{row_label(row)}: max vector diff {diff:.2e} > {ATOL}")
    return failures

def latest_per_setting(rows):
    latest = {}
    for row in rows:
        latest[(row["mode"], int(row["k"]))] = row
    return latest

def plot(groups, png_path):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, ((docs, limit), rows) in zip(axes[0], sorted(groups.items())):
        latest = latest_per_setting(rows)
        for mode in sorted({mode for mode, _k in latest}):
            ks = sorted(k for m, k in latest if m == mode)
            ax.plot(ks, [float(latest[(mode, k)]["run_s"]) for k in ks], marker="o", label=mode)
            ax.set_xticks(ks)
        ax.set_title(f"{Path(docs).stem}, limit={limit}")
        ax.set_xlabel("k (cores)")
        ax.set_ylabel("run_s (chunk + embed)")
        ax.legend()
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    png_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PNG
    groups = group_by_dataset(read_rows(csv_path))
    if not groups:
        sys.exit(f"no rows in {csv_path}")

    failures = [
        f"[{docs} limit={limit}] {failure}"
        for (docs, limit), rows in groups.items()
        for failure in check_group((docs, limit), rows)
    ]
    if failures:
        sys.exit("equivalence gate failed, not plotting:\n  " + "\n  ".join(failures))

    plot(groups, png_path)
    print(f"gate passed for {len(groups)} dataset(s); wrote {png_path}")

if __name__ == "__main__":
    main()
