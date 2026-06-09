# PiPNN Phase-Isolated Bench Suite

Four bench targets under this directory:

| Target           | Kind      | Measures                                                    |
| ---------------- | --------- | ----------------------------------------------------------- |
| `partition`      | criterion | `partition::partition` (RBC) wall time                   |
| `leaf_build`     | criterion | `leaf_build::build_leaf` iterated over cached leaves        |
| `final_prune`    | criterion | `builder::final_prune_from_candidates` (RobustPrune) time   |
| `phase_profile`  | stdout    | All 3 phases with sub-step breakdowns (feature-gated)       |

Four workloads, dim × scale:

| Name          | npoints    | ndims | metric               | expected file             |
| ------------- | ---------- | ----- | -------------------- | ------------------------- |
| `bigann_1m`   | 1,000,000  | 128   | `squared_l2`         | `bigann_1m_fp16.bin`      |
| `bigann_10m`  | 10,000,000 | 128   | `squared_l2`         | `bigann_10m_fp16.bin`     |
| `enron_1m`    | 1,087,932  | 384   | `cosine_normalized`  | `enron_fp16.bin`          |
| `enron_10m`   | 10,000,000 | 384   | `cosine_normalized`  | `enron_10m_fp16.bin`      |

All loads are fp16, run on a fixed 16-thread rayon pool, with the canonical
per-load config from `CLAUDE.md` (c_max=256, fanout=[8,3], leaf_k=2, l_max=64,
p_samp=0.005; `num_hash_planes`=12 for BigANN, 14 for Enron).

---

## Dataset setup

Dataset files are looked up in this order:

1. `$DISKANN_DATASETS` environment variable (directory)
2. `./datasets/` (relative to the current working directory)

Missing files **skip the affected load** with a `[skip]` message — benches
never panic on missing data, so you can run partial matrices.

### File format

Standard DiskANN fp16 binary:

```
offset 0:  u32 LE  npoints
offset 4:  u32 LE  ndims
offset 8:  [f16]   npoints * ndims values, row-major
```

Each 2 bytes = one `f16`. The bench harness `mmap`s the file and exposes a
zero-copy `&[f16]` view — no copy or conversion on load.

### Pointing at existing datasets

```bash
# Single run
DISKANN_DATASETS=/path/to/datasets cargo bench -p diskann-pipnn --bench partition

# Or export once
export DISKANN_DATASETS=/path/to/datasets
cargo bench -p diskann-pipnn --bench partition
```

---

## Running the criterion benches

```bash
# Default: 1M loads only (~seconds per bench)
cargo bench -p diskann-pipnn --bench partition
cargo bench -p diskann-pipnn --bench leaf_build
cargo bench -p diskann-pipnn --bench final_prune

# 10M loads (~15 min per bench; criterion uses sample_size=10, measurement_time=90s)
PIPNN_BENCH_SCALE=10m cargo bench -p diskann-pipnn --bench partition

# All 4 loads (1M + 10M)
PIPNN_BENCH_SCALE=all cargo bench -p diskann-pipnn --bench leaf_build

# Run one specific load by criterion name filter
cargo bench -p diskann-pipnn --bench partition -- bigann_1m
```

`PIPNN_BENCH_SCALE` values:
- `1m` (default) — only 1M loads
- `10m` — only 10M loads
- `all` — all 4 loads

Output is criterion's standard — stdout summaries plus HTML reports under
`target/criterion/`.

---

## Running the sub-step profiler

Requires the `bench-profiling` Cargo feature (off by default so production
builds are byte-identical to unprofiled builds — verified: zero `PhaseTimer`
symbols in the default release artifact).

```bash
# One load at a time; runs partition → leaf_build → final_prune and prints
# a stdout table of sub-step timings per phase.
cargo bench -p diskann-pipnn --bench phase_profile \
    --features bench-profiling -- --load bigann_1m

cargo bench -p diskann-pipnn --bench phase_profile \
    --features bench-profiling -- --load enron_10m
```

Sample output:

```
Dataset: enron_1m (1087932 x 384, CosineNormalized)
Config:  c_max=256 c_min=16 fanout=[8, 3] leaf_k=2 l_max=64 hp=14 p_samp=0.005

=== Partition (RBC) — 5234 leaves (3.142s wall) ===
  sub-step                            time(s)        %
  partition/leader_sample               0.021     0.7%
  partition/assign_to_leaders           2.981    94.9%
  partition/merge_small                 0.124     4.0%
  TOTAL (summed sub-steps)              3.126   100.0%

=== Leaf build — 38.5M edges across 5234 leaves (2.814s wall) ===
  sub-step                            time(s)        %
  leaf_build/bidir                      0.382    13.6%
  leaf_build/conv                       0.061     2.2%
  leaf_build/dist                       0.163     5.8%
  leaf_build/gemm                       1.854    65.9%
  leaf_build/knn                        0.279     9.9%
  leaf_build/norms                      0.032     1.1%
  leaf_build/seen_fill                  0.043     1.5%
  TOTAL (summed sub-steps)              2.814   100.0%

=== Final prune (RobustPrune) — avg degree 47.3 → 28.1 (1.982s wall) ===
  sub-step                            time(s)        %
  final_prune/occlusion_loop            1.541    77.8%
  final_prune/prune_path                0.440    22.2%
  TOTAL (summed sub-steps)              1.981   100.0%
```

The sub-step names map directly to source-code locations:
- `partition/*` — `src/partition.rs::partition_one_level`
- `leaf_build/*` — `src/leaf_build.rs::build_leaf_with_buffers`
- `final_prune/*` — `src/builder.rs::final_prune_from_candidates`

---

## Fixture caching

**Leaves** (partition output) are cached in-process, rebuilt once per `cargo bench`
invocation. Cost: ~6s on 1M, ~60s on 10M.

**Candidate graph** (partition + leaf_build + HashPrune::extract_graph_for_prune)
is disk-cached at:

```
$CARGO_TARGET_DIR/bench-fixtures/<load>-<config_hash>.candidates
```

First `final_prune` bench run on a new load pays the full ~90s build; later
runs load the fixture in ~1s. The cache is invalidated automatically when
any of the following change: source data path, source file mtime, `c_max`,
`c_min`, `p_samp`, `fanout`, `k`, `l_max`, `num_hash_planes`, `metric`.

To force a full rebuild, delete the cache file:

```bash
rm -rf target/bench-fixtures/
```

The cache size is `npoints × l_max × 8` bytes + a small offset table —
~5 GB for `bigann_10m` at l_max=64.

---

## Typical workflow

- **"Did my change help?"** — run the affected criterion bench on `PIPNN_BENCH_SCALE=1m`
  (fast, statistical). Criterion stores a baseline so re-runs show regressions.
- **"Where is the time going inside this phase?"** — run the profiler with
  `--features bench-profiling --load <load>`. The sub-step table tells you
  which inner loop to optimize.
- **Pre-merge validation** — run all 4 criterion benches at `PIPNN_BENCH_SCALE=all`
  (~1 hour total) to confirm your change doesn't regress 10M.

The profiler feature is diagnostic-only and adds ~50 ns per `PhaseTimer` drop
(per-thread parking_lot mutex + HashMap insert). For hot inner loops like
`final_prune/occlusion_loop` that run tens of millions of times, the observer
effect is non-trivial — treat profiler sub-step numbers as *relative* cost
attribution, not absolute latency.
