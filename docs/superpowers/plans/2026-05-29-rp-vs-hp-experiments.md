# RobustPrune × HashPrune Experiment Harness — Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans (in-session) to implement. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether Vamana's RobustPrune beats PiPNN's HashPrune for graph quality (PiPNN Exp 1 & 2), and whether HashPrune can replace RobustPrune inside Vamana's insert (Exp 3), on Enron 1M (384d, cosine_normalized), reported as build time / avg_degree / peak RSS / recall@1000 + QPS at L∈{1000,1500,2000,2500,3000}.

**Architecture:** Reuse the existing JSON-driven build→search→recall/QPS harness (`diskann-benchmark`). Exp 1 & 2 are new PiPNN leaf-build + merge modes selected via a new `leaf_prune_mode` field on `BuildAlgorithm::PiPNN`, dispatched inside `diskann-pipnn::builder`. Exp 3 is a new `prune_mode` on `BuildAlgorithm::Vamana` backed by a self-contained reservoir module inside the `diskann` crate (feature-gated, no dependency on diskann-pipnn). All prunes reuse the existing `final_prune_from_candidates` RobustPrune (occlude_list port, DiskANN iterative-alpha mode).

**Tech Stack:** Rust, rayon, parking_lot; `diskann-pipnn`, `diskann`, `diskann-disk`, `diskann-benchmark` crates; mimalloc; AVX2 (WSL).

**Worktree:** `.worktrees/pipnn-rp-hp-exp` (branch `pipnn-rp-hp-exp`, baseline = snapshot of `pipnn-scaling-opt` working tree). Datasets symlinked at `./datasets`.

**Prior art reused (logic, adapted to current API):**
- `.worktrees/pipnn-robustprune/diskann-pipnn/src/candidate_pool.rs` — `CandidatePool` (bounded, l_max farthest-eviction) + `AppendOnlyPool` (unbounded), both → `extract_dedupped_sorted() -> Vec<Vec<(u32,f32)>>`.

**Build:** `RUSTFLAGS='-C target-cpu=native' cargo build --release -p diskann-benchmark --features 'diskann-disk/pipnn,disk-index'` (+ `,diskann/hashprune-build` for Exp 3).

**WSL safety:** 31 GB RAM / 16 cores. Every benchmark run goes through `scripts/run_exp.sh` which wraps the binary in an RSS watchdog that kills the process if RSS exceeds 26 GB. Runs are sequential.

---

## Variant → config matrix (what each run sets)

| Run | algorithm | key fields |
|---|---|---|
| base-pipnn | PiPNN | `leaf_prune_mode:"Baseline"`, `leaf_k:3`, `final_prune:false` |
| base-pipnn-fp | PiPNN | `leaf_prune_mode:"Baseline"`, `leaf_k:3`, `final_prune:true` |
| base-vamana | Vamana | `prune_mode:"RobustPrune"` (default), R=64 L=72 |
| **exp1** | PiPNN | `leaf_prune_mode:"RobustNoGemm"`, `merge_l_max:256` |
| **exp2** | PiPNN | `leaf_prune_mode:"GemmTopKRobust"`, `leaf_k:128`, `merge_l_max:256` |
| **exp3** | Vamana | `prune_mode:"HashpruneReservoir"`, `num_hash_planes:14`, `l_max:64` |

`merge_l_max:0` ⇒ unbounded `AppendOnlyPool` (faithful "keep all", run behind watchdog).

---

## File Structure

**diskann-pipnn (Exp 1 & 2):**
- Create `diskann-pipnn/src/candidate_pool.rs` — port of CandidatePool + AppendOnlyPool, with an `add_edges_batched(&[Edge])` method mirroring `HashPrune::add_edges_batched`.
- Modify `diskann-pipnn/src/leaf_build.rs` — add `build_leaf_robust_no_gemm(...)` (Exp 1) and `build_leaf_gemm_topk_robust(...)` (Exp 2); both return `Vec<Edge>` already RobustPruned-to-max_degree per source.
- Modify `diskann-pipnn/src/lib.rs` — add `LeafPruneMode` enum + `leaf_prune_mode` + `merge_l_max` fields to `PiPNNConfig`; expose a reusable `robust_prune_candidates(...)` (factored from `final_prune_from_candidates`).
- Modify `diskann-pipnn/src/builder.rs` — dispatch on `leaf_prune_mode`: Baseline → HashPrune (unchanged); Exp1/Exp2 → new leaf fn + CandidatePool/AppendOnlyPool merge → `robust_prune_candidates` final pass.

**diskann-disk (config surface):**
- Modify `diskann-disk/src/build/configuration/build_algorithm.rs` — add `leaf_prune_mode` + `merge_l_max` to `PiPNN{}`; add `prune_mode` to `Vamana` (struct variant w/ defaults, backward-compatible); map in `to_pipnn_config`.

**diskann (Exp 3):**
- Create `diskann/src/graph/hashprune_reservoir.rs` — self-contained LSH sketch + per-node reservoir (mirrors diskann-pipnn HashPrune; no external dep). Feature `hashprune-build`.
- Modify `diskann/src/graph/index.rs` — in `search_and_prune` / `prune_back_edges`, when reservoir mode active, route candidate pool → reservoir → adjacency instead of `robust_prune`; final RobustPrune pass via existing `prune_range`.
- Modify `diskann/Cargo.toml` — add `hashprune-build` feature.
- Plumb `prune_mode` from `BuildAlgorithm::Vamana` → diskann index `Config` (diskann-disk build path).

**Harness:**
- Create `scripts/run_exp.sh` — RSS watchdog wrapper + result parser.
- Create `experiments/*.json` — one config per run (table above).
- Create `experiments/RESULTS.md` — consolidated comparison table (filled as runs complete).

---

## Phase 0 — Scaffold + baselines (uses the already-building stock binary)

### Task 0.1: RSS watchdog runner
**Files:** Create `scripts/run_exp.sh`
- [ ] Write a script: args `<config.json> <label>`; launches `target/release/diskann-benchmark <config.json>` under `/usr/bin/time -v`, tees stdout to `experiments/logs/<label>.log`, and runs a background poller that `kill -9`s the run if `VmRSS` (from `/proc/<pid>/status`) exceeds 26 GB. On exit, greps the log for build time, recall, QPS, peak RSS and appends a row to `experiments/RESULTS.md`.
- [ ] `chmod +x scripts/run_exp.sh`; smoke-test arg parsing with `--help`.
- [ ] Commit.

### Task 0.2: Baseline configs
**Files:** Create `experiments/base_pipnn.json`, `experiments/base_pipnn_fp.json`, `experiments/base_vamana.json`
- [ ] Author the three JSONs (Enron 1M, dim 384, cosine_normalized, num_pq_chunks 192, max_degree 64, search_list [1000,1500,2000,2500,3000], recall_at 1000, beam_width 4, num_threads 16). PiPNN: c_max 256 c_min 16 fanout [8,3] l_max 64 num_hash_planes 14. Vamana: l_build 72.
- [ ] Validate each parses: `target/release/diskann-benchmark --dry-run` if supported, else rely on Task 0.3.
- [ ] Commit.

### Task 0.3: Run baselines (validate harness end-to-end on this WSL)
- [ ] After build completes: `scripts/run_exp.sh experiments/base_pipnn.json base-pipnn`. Confirm it builds + searches + emits recall/QPS. Verify peak RSS < 6 GB.
- [ ] Run `base-pipnn-fp` and `base-vamana` sequentially.
- [ ] Sanity-check numbers vs CLAUDE.md Enron 1M ballpark; record in RESULTS.md.
- [ ] Commit RESULTS.md.

**GATE:** Baselines reproduce sane recall/QPS before writing any experiment code.

---

## Phase 1 — Reusable RobustPrune helper

### Task 1.1: Factor `robust_prune_candidates` out of `final_prune_from_candidates`
**Files:** Modify `diskann-pipnn/src/lib.rs`; Test: inline `#[cfg(test)]`
- [ ] Extract the per-node occlusion body of `final_prune_from_candidates` into `pub fn robust_prune_candidates(node_f32:&[f32], cand_ids:&[u32], cand_f32:&[f32], nc:usize, ndims:usize, dist_fn:&..., max_degree:usize, alpha:f32, saturate:bool, iterative:bool) -> Vec<u32>` (the existing fn becomes a thin parallel wrapper). `iterative` selects DiskANN incremental-alpha vs paper single-pass (mirrors `PIPNN_DISKANN_PRUNE`).
- [ ] Write failing unit test `test_robust_prune_occludes_collinear`: 3 collinear points, assert the middle is occluded at alpha=1.0.
- [ ] Run `cargo test -p diskann-pipnn robust_prune_occludes -- --nocapture` → FAIL.
- [ ] Implement; re-run → PASS. Confirm `final_prune_from_candidates` output unchanged via existing `test_recall`.
- [ ] Commit.

---

## Phase 2 — Candidate accumulators (port prior art)

### Task 2.1: Port CandidatePool + AppendOnlyPool
**Files:** Create `diskann-pipnn/src/candidate_pool.rs`; register `pub mod candidate_pool;` in lib.rs
- [ ] Port both structs. Replace the old CSR `add_edges_grouped(...)` with `add_edges_batched(&self, edges:&[crate::leaf_build::Edge])` (group by `edge.src`, lock once per source — mirror `HashPrune::add_edges_batched`). Replace `collect_installed()` (old `rayon_util`) with `.collect()` (callers already run inside an installed pool).
- [ ] `CandidatePool::new(npoints, l_max)`, `AppendOnlyPool::new(npoints)`, both `extract_dedupped_sorted(self) -> Vec<Vec<(u32,f32)>>`.
- [ ] Unit tests: `test_candidate_pool_dedup_keeps_min` (insert (5,2.0),(5,1.0),(7,3.0) → [(5,1.0),(7,3.0)] sorted), `test_candidate_pool_bounded_evicts_farthest` (l_max=2, three inserts → keeps 2 closest), `test_append_only_unbounded`.
- [ ] `cargo test -p diskann-pipnn candidate_pool` → PASS.
- [ ] Commit.

---

## Phase 3 — Leaf builders (Exp 1 & 2)

### Task 3.1: Exp 1 leaf — all-pairs, no GEMM, RobustPrune to max_degree
**Files:** Modify `diskann-pipnn/src/leaf_build.rs`
- [ ] Add `build_leaf_robust_no_gemm<T>(data, ndims, indices, max_degree, metric, alpha, dist_fn) -> Vec<Edge>`. For each local point i: compute distance to every other leaf-mate via direct `dist_fn.call` (no GEMM, no matrix; convert each pair on the fly or use a per-leaf f32 gather buffer like `build_leaf_with_buffers` does for conversion), build candidate list `(global_id, dist)`, sort ascending, call `robust_prune_candidates(..., max_degree, alpha, saturate=false, iterative=true)`, emit one `Edge{src=global_i, dst, distance}` per kept neighbor. (No bidirection here — the accumulator + final prune handle symmetry; matches "keep those candidates per point".)
- [ ] Unit test `test_leaf_robust_no_gemm_diverse`: small leaf, assert kept set ⊆ leaf, size ≤ max_degree, and occlusion removed a redundant near-duplicate.
- [ ] `cargo test -p diskann-pipnn leaf_robust_no_gemm` → PASS.
- [ ] Commit.

### Task 3.2: Exp 2 leaf — GEMM top-128 then RobustPrune to max_degree
**Files:** Modify `diskann-pipnn/src/leaf_build.rs`
- [ ] Add `build_leaf_gemm_topk_robust<T>(data, ndims, indices, leaf_k, max_degree, metric, alpha, bufs) -> Vec<Edge>`: reuse the GEMM all-pairs + `extract_knn(dist_matrix, n, leaf_k)` path (leaf_k=128) to get top-leaf_k candidates per point, then `robust_prune_candidates(... max_degree ...)` on those candidates, emit Edges. (Distances from the GEMM matrix; convert to the f32 gather already in `bufs.local_data` for the prune.)
- [ ] Unit test `test_leaf_gemm_topk_robust`: leaf size 40, leaf_k=20, max_degree 8 → each source ≤ 8 neighbors, all within leaf.
- [ ] `cargo test -p diskann-pipnn gemm_topk_robust` → PASS.
- [ ] Commit.

---

## Phase 4 — Config + builder dispatch

### Task 4.1: Config fields
**Files:** Modify `diskann-pipnn/src/lib.rs`, `diskann-disk/.../build_algorithm.rs`
- [ ] Add `pub enum LeafPruneMode { Baseline, RobustNoGemm, GemmTopKRobust }` (serde, default Baseline) + `leaf_prune_mode: LeafPruneMode` + `merge_l_max: usize` (default 256) to `PiPNNConfig`; extend `validate()`.
- [ ] Add same two fields to `BuildAlgorithm::PiPNN{}` (serde defaults) + map in `to_pipnn_config`.
- [ ] Update the 3 builder tests that construct `PiPNNConfig{..}` literally (compile fix).
- [ ] `cargo build -p diskann-disk --features pipnn` → OK; serde roundtrip test passes.
- [ ] Commit.

### Task 4.2: Builder dispatch
**Files:** Modify `diskann-pipnn/src/builder.rs` (`build_internal_impl`)
- [ ] Branch on `config.leaf_prune_mode`:
  - `Baseline` → existing HashPrune path (unchanged).
  - `RobustNoGemm` / `GemmTopKRobust` → per-leaf call the matching new leaf fn (Task 3.x) producing pruned Edges; stream into `CandidatePool::new(npoints, merge_l_max)` (or `AppendOnlyPool` if `merge_l_max==0`) via `add_edges_batched`; after all leaves → `extract_dedupped_sorted()` → `final_prune_from_candidates(... max_degree, alpha, saturate=true)`.
- [ ] Keep timing stats populated (leaf_build_secs, final_prune_secs).
- [ ] Reuse existing `test_recall` shape: add `test_build_exp1_recall` + `test_build_exp2_recall` on random data asserting avg_degree>0, recall above a floor.
- [ ] `cargo test -p diskann-pipnn build_exp` → PASS.
- [ ] Commit.

---

## Phase 5 — Run Exp 1 & 2

### Task 5.1: Configs + runs
**Files:** Create `experiments/exp1.json`, `experiments/exp2.json`
- [ ] Author from base_pipnn.json + the matrix above.
- [ ] Rebuild release binary.
- [ ] `scripts/run_exp.sh experiments/exp1.json exp1` (watch RSS).
- [ ] `scripts/run_exp.sh experiments/exp2.json exp2`.
- [ ] (Optional, behind watchdog) `merge_l_max:0` unbounded variants exp1-unb / exp2-unb.
- [ ] Append rows to RESULTS.md; commit.

**GATE:** Exp 1 & 2 numbers captured + sanity-checked vs baselines before starting Exp 3.

---

## Phase 6 — Exp 3: full reservoir-based Vamana insert (highest risk)

### Task 6.1: Investigation spike (no behavior change)
**Files:** none (notes appended to this plan)
- [ ] Trace `BuildAlgorithm::Vamana` → diskann index `Config` construction in `diskann-disk/src/build/builder/build.rs` (the `else` branch). Identify where to inject `prune_mode`.
- [ ] In `diskann/src/graph/index.rs`, confirm the two `robust_prune` call sites (`search_and_prune` ~540, `prune_back_edges` ~759) and how `new_out_neighbors` / adjacency are written; confirm where a post-build full prune (`prune_range`, ~2968) can run.
- [ ] Decide reservoir keying: per-node LSH sketch computed lazily from the node's vector via the prune accessor's distance/vector access. Document the exact hook in this plan; STOP and re-confirm scope if the generic/async signatures require touching >3 functions.

### Task 6.2: Reservoir module (feature-gated, self-contained)
**Files:** Create `diskann/src/graph/hashprune_reservoir.rs`; `diskann/Cargo.toml` feature `hashprune-build`
- [ ] Port the LSH sketch (`relative_hash`) + `HashPruneReservoir` (insert/evict/get_neighbors_saturated) from diskann-pipnn, dependency-free. `npoints`-sized `Vec<Mutex<Reservoir>>` + lazily-filled sketch store.
- [ ] Unit tests mirroring diskann-pipnn reservoir tests (`insert evicts farthest`, `same hash keeps closer`).
- [ ] `cargo test -p diskann --features hashprune-build hashprune_reservoir` → PASS.
- [ ] Commit.

### Task 6.3: Wire into insert + final prune
**Files:** Modify `diskann/src/graph/index.rs`; plumb `prune_mode` (diskann-disk)
- [ ] Add `prune_mode` to `BuildAlgorithm::Vamana` + diskann index `Config`; thread to the build.
- [ ] When `HashpruneReservoir`: replace each `robust_prune` candidate-selection with reservoir insert of the candidate pool, then materialize the node's adjacency from `get_neighbors_saturated(max_degree)` (so greedy search sees edges). Back-edges similarly push into the target's reservoir + re-materialize.
- [ ] After build completes, run one `prune_range` RobustPrune pass over all nodes (genuine diskann prune).
- [ ] Validate on a tiny in-memory build test: graph builds, non-zero avg degree, search returns sane neighbors.
- [ ] `cargo build --release -p diskann-benchmark --features 'diskann-disk/pipnn,disk-index,diskann/hashprune-build'` → OK.
- [ ] Commit.

### Task 6.4: Run Exp 3
**Files:** Create `experiments/exp3.json`
- [ ] `scripts/run_exp.sh experiments/exp3.json exp3` (watch RSS).
- [ ] Append to RESULTS.md; commit.

---

## Phase 7 — Consolidated report
- [ ] Fill `experiments/RESULTS.md`: one table, all 6 runs × {build s, avg_degree, peak RSS, recall@1000 + QPS at each L}. Add a short analysis: does RobustPrune-merge beat HashPrune (Exp1/2 vs base-pipnn / base-pipnn-fp)? Does HashPrune speed up Vamana at comparable recall (Exp3 vs base-vamana)?
- [ ] Commit. Surface RESULTS.md to the user.

---

## Risks / notes
- **Exp 2 memory:** per-leaf prune-to-max_degree bounds accumulation to ~fanout×max_degree (≈1536) pre-dedup; `merge_l_max:256` caps to ~2 GB at 1M. Unbounded only behind watchdog.
- **Exp 3 invasiveness:** diskann insert is async + generic. Task 6.1 is a hard gate — if wiring touches >3 functions or fights the type system, re-surface scope before proceeding.
- **"diskann path code":** PiPNN-side prunes use `final_prune_from_candidates` (occlude_list port, iterative-alpha). Exp 3 final prune uses genuine in-crate `prune_range`.
- **Enron data symlink** resolves into `.worktrees/turboquant/...` — verified present; if a future cleanup removes it, repoint `datasets/enron_fp16.bin`.
