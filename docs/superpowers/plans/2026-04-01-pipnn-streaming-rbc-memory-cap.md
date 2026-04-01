# PiPNN Streaming RBC Memory-Cap Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the hidden over-budget PiPNN fallback as a whole-dataset, streaming RBC partition/build path that respects the build memory cap while preserving current PiPNN semantics and reusing existing PiPNN code.

**Architecture:** Keep current one-shot PiPNN unchanged when the estimated peak fits the build memory limit. When the estimate exceeds the limit, do **not** switch to Vamana-style shard/merge. Instead, keep one logical PiPNN build and replace the memory-heavy whole-dataset partition materialization with a streaming RBC pipeline over the full dataset, then reuse the existing leaf build and HashPrune merge path. `final_prune=false` is the primary supported streaming target.

**Tech Stack:** Rust, diskann-pipnn, diskann-disk, Rayon, existing PiPNN RBC partitioning, existing PiPNN leaf GEMM builder, existing PiPNN HashPrune.

---

## Non-Negotiable Guardrails

- [ ] **Guardrail 1: Do not implement Vamana-style data shards for PiPNN**

This plan is **not** “split dataset into shard files, build independent PiPNN graphs, then merge shard graphs.”

The corrected design is:

1. full dataset remains one logical PiPNN problem
2. RBC-style partitioning remains the partitioning model
3. over-budget mode streams the RBC work to reduce peak RAM
4. leaf edges still flow into one logical HashPrune merge
5. user-facing API remains a hidden automatic switch based on estimated RAM vs limit

- [ ] **Guardrail 2: Reuse current PiPNN semantics wherever possible**

Keep these semantics unless a test or explicit design constraint says otherwise:

1. current medoid behavior
2. current HashPrune merge behavior
3. current leaf-local bi-directed k-NN build behavior
4. current save format
5. `final_prune=false` as the primary over-budget path

- [ ] **Guardrail 3: TDD only**

No production-code changes for the streaming path before the failing tests are written and run.

---

## Current Code Map

### Files already in play

- Modify: `diskann-pipnn/src/builder.rs`
- Modify: `diskann-pipnn/src/partition.rs`
- Modify: `diskann-disk/src/build/builder/build.rs`
- Modify: `diskann-disk/src/build/configuration/build_algorithm.rs`
- Possibly modify: `diskann-pipnn/src/hash_prune.rs`
- Review only unless needed: `diskann-pipnn/src/leaf_build.rs`

### Existing seams to preserve

- `diskann-pipnn/src/builder.rs`
  - `build_internal_impl(...)` is the current one-shot reference path
  - `choose_typed_build_plan(...)` is the current hidden RAM gate seam
  - `build_typed_sharded_scaffold(...)` is the current placeholder that must become the real over-budget path

- `diskann-pipnn/src/partition.rs`
  - `parallel_partition(...)` and `parallel_partition_quantized(...)` currently produce fully materialized leaves
  - `partition_assign(...)` already stripes assignment work and is the best reuse seam for streaming RBC

- `diskann-pipnn/src/hash_prune.rs`
  - current merge path already supports batched edge insertion
  - this is the semantic sink for streamed leaf output

- `diskann-disk/src/build/builder/build.rs`
  - `build_pipnn_index_sync()` is the integration dispatch point for the hidden one-shot vs over-budget switch

---

## Task 1: Lock the corrected design with failing tests

**Files:**
- Modify: `diskann-disk/src/build/configuration/build_algorithm.rs`
- Modify: `diskann-pipnn/src/partition.rs`
- Possibly modify: `diskann-pipnn/src/builder.rs`

- [ ] **Step 1: Write the failing hidden-switch test**

Add a test in `diskann-disk/src/build/configuration/build_algorithm.rs` that asserts PiPNN over-budget selection means “PiPNN streaming/over-budget path” semantics, not generic merged-Vamana sharding semantics.

Suggested test name:

```rust
fn test_pipnn_over_budget_selects_streaming_rbc_not_vamana_merged_path()
```

- [ ] **Step 2: Run only that test and verify it fails for the right reason**

Run:

```bash
cargo test -p diskann-disk --features pipnn test_pipnn_over_budget_selects_streaming_rbc_not_vamana_merged_path
```

Expected now: FAIL because the current code only knows one-shot vs `NeedsSharded` scaffold / merged-shard semantics, not the corrected streaming-RBC distinction.

- [ ] **Step 3: Write the failing partition-semantics test**

Add a test in `diskann-pipnn/src/partition.rs` proving that the over-budget design is still whole-dataset RBC semantics across artificial read-chunk boundaries, rather than independent data-shard partitioning.

Suggested test name:

```rust
fn test_streaming_rbc_partition_preserves_global_overlap_across_input_chunks()
```

The test should assert that points from different artificial input chunks can still land in overlapping/global RBC leaves.

- [ ] **Step 4: Run only that test and verify it fails for the right reason**

Run:

```bash
cargo test -p diskann-pipnn test_streaming_rbc_partition_preserves_global_overlap_across_input_chunks
```

Expected now: FAIL because there is no streaming whole-dataset RBC entry point yet.

- [ ] **Step 5: Commit nothing yet**

Do not commit until the first green slice is implemented and verified.

---

## Task 2: Introduce explicit internal plan types for one-shot vs over-budget streaming PiPNN

**Files:**
- Modify: `diskann-pipnn/src/builder.rs`
- Modify: `diskann-disk/src/build/builder/build.rs`

- [ ] **Step 1: Replace ambiguous naming**

Rename or replace internal terms that imply Vamana-style sharding.

Preferred direction:

```rust
pub enum PiPNNBuildPlan {
    OneShot { estimated_peak_bytes: usize },
    StreamingRbc { estimated_peak_bytes: usize },
}
```

Avoid `NeedsSharded` for the steady-state design because it encourages drift back to the wrong architecture.

- [ ] **Step 2: Run the smallest plan-selection test and verify it still fails or needs updates**

Run:

```bash
cargo test -p diskann-pipnn test_choose_typed_build_plan_marks_sharded_when_over_budget
```

Update the existing test names and expectations to the corrected terminology.

- [ ] **Step 3: Implement minimal plan-selection changes only**

Make the planner return the corrected over-budget internal plan type without yet implementing the full streaming build.

- [ ] **Step 4: Re-run targeted planner tests**

Run:

```bash
cargo test -p diskann-pipnn test_choose_typed_build_plan_marks_one_shot_when_within_budget
cargo test -p diskann-pipnn test_choose_typed_build_plan_marks_streaming_when_over_budget
```

Expected: PASS.

---

## Task 3: Add a streaming whole-dataset RBC partition entry point

**Files:**
- Modify: `diskann-pipnn/src/partition.rs`
- Test: `diskann-pipnn/src/partition.rs`

- [ ] **Step 1: Design the smallest streaming partition abstraction**

Add an internal API that keeps the current RBC logic but allows the whole-dataset partition work to be emitted incrementally instead of requiring the final full leaf set up front.

Preferred shape:

```rust
pub(crate) fn stream_partition<T, F>(...) -> PiPNNResult<()>
where
    F: FnMut(Leaf) -> PiPNNResult<()>
```

or another minimal internal callback/iterator form with the same effect.

- [ ] **Step 2: Reuse current RBC pieces instead of rewriting partition logic**

Reuse:

1. leader sampling
2. assignment logic
3. undersized-cluster merging
4. recursive split behavior
5. c_max / c_min / fanout semantics

Do not introduce a separate “shard partition algorithm.”

- [ ] **Step 3: Implement the smallest version needed to make the new partition test pass**

The first green goal is not full build integration. The first green goal is proving the whole-dataset streaming RBC seam exists and preserves global overlap semantics.

- [ ] **Step 4: Run the new partition test and nearby partition tests**

Run:

```bash
cargo test -p diskann-pipnn test_streaming_rbc_partition_preserves_global_overlap_across_input_chunks
cargo test -p diskann-pipnn test_partition_overlap
cargo test -p diskann-pipnn test_partition_respects_c_max
```

Expected: PASS.

---

## Task 4: Replace the scaffold error with a real streaming PiPNN build path

**Files:**
- Modify: `diskann-pipnn/src/builder.rs`
- Possibly modify: `diskann-pipnn/src/hash_prune.rs`
- Review: `diskann-pipnn/src/leaf_build.rs`

- [ ] **Step 1: Write the failing builder-semantics test**

Add a test in `diskann-pipnn/src/builder.rs` that compares the new over-budget streaming path to the current one-shot path on a deterministic small dataset with `final_prune=false`.

Suggested test name:

```rust
fn test_streaming_rbc_build_matches_one_shot_semantics_when_final_prune_is_off()
```

Start with invariant-level checks if exact adjacency equality is too brittle:

1. same `npoints`
2. same `medoid`
3. no isolated-node regression
4. degree bounded by `max_degree`
5. same graph save/load validity

- [ ] **Step 2: Run that test and verify it fails because the streaming path is still not implemented**

Run:

```bash
cargo test -p diskann-pipnn test_streaming_rbc_build_matches_one_shot_semantics_when_final_prune_is_off
```

- [ ] **Step 3: Implement the real over-budget builder path**

In `diskann-pipnn/src/builder.rs`, replace the placeholder error path with this structure:

1. compute medoid once
2. initialize HashPrune once
3. for each replica
4. stream whole-dataset RBC leaves
5. for each emitted leaf, call existing leaf builder
6. immediately push edges into existing HashPrune batched merge
7. after all replicas, extract graph normally
8. keep `final_prune=false` as the supported over-budget path

Do not create independent per-shard graphs.

- [ ] **Step 4: If needed, add a tiny HashPrune/order test**

Only if the builder test exposes ordering sensitivity, add:

```rust
fn test_streamed_leaf_batches_produce_same_hashprune_graph_as_materialized_leaf_batches()
```

Keep this focused on the builder/HashPrune seam, not a broad algorithm benchmark.

- [ ] **Step 5: Run targeted builder tests**

Run:

```bash
cargo test -p diskann-pipnn test_streaming_rbc_build_matches_one_shot_semantics_when_final_prune_is_off
cargo test -p diskann-pipnn test_build_small
cargo test -p diskann-pipnn test_shard_edge_batches_match_monolithic_hash_prune
```

Expected: PASS.

---

## Task 5: Integrate the corrected hidden switch into disk build dispatch

**Files:**
- Modify: `diskann-disk/src/build/builder/build.rs`
- Possibly modify: `diskann-disk/src/build/builder/tests.rs`

- [ ] **Step 1: Write the failing disk-builder dispatch test**

Add a targeted integration test around `build_pipnn_index_sync()` dispatch.

Suggested test name:

```rust
fn test_disk_builder_dispatches_over_budget_pipnn_to_streaming_rbc_path()
```

This test should prove the hidden switch is runtime behavior, not just a pure helper.

- [ ] **Step 2: Run the test and verify it fails before the dispatch change**

Run:

```bash
cargo test -p diskann-disk --features pipnn test_disk_builder_dispatches_over_budget_pipnn_to_streaming_rbc_path
```

- [ ] **Step 3: Implement minimal dispatch wiring**

Update `build_pipnn_index_sync()` so that:

1. one-shot budget fit => existing one-shot PiPNN path
2. over-budget => new streaming whole-dataset RBC PiPNN path
3. no user-facing shard mode is introduced

- [ ] **Step 4: Re-run the dispatch test plus the existing PiPNN budget-switch test**

Run:

```bash
cargo test -p diskann-disk --features pipnn test_disk_builder_dispatches_over_budget_pipnn_to_streaming_rbc_path
cargo test -p diskann-disk --features pipnn test_pipnn_build_plan_switches_to_sharded_when_over_budget
```

Rename the second test to corrected terminology as part of this slice.

---

## Task 6: Verification and cleanup

**Files:**
- Review: `diskann-pipnn/src/partition.rs`
- Review: `diskann-pipnn/src/builder.rs`
- Review: `diskann-disk/src/build/builder/build.rs`
- Review: `diskann-disk/src/build/builder/core.rs`
- Review: `diskann-disk/src/build/configuration/build_algorithm.rs`

- [ ] **Step 1: Run targeted PiPNN tests**

Run:

```bash
cargo test -p diskann-pipnn test_streaming_rbc_build_matches_one_shot_semantics_when_final_prune_is_off
cargo test -p diskann-pipnn test_streaming_rbc_partition_preserves_global_overlap_across_input_chunks
cargo test -p diskann-pipnn test_build_small
cargo test -p diskann-pipnn test_search_basic
```

- [ ] **Step 2: Run targeted disk-builder tests**

Run:

```bash
cargo test -p diskann-disk --features pipnn test_disk_builder_dispatches_over_budget_pipnn_to_streaming_rbc_path
cargo test -p diskann-disk --features pipnn test_to_pipnn_config_pipnn_returns_some
```

- [ ] **Step 3: Build the affected crates**

Run:

```bash
cargo build -p diskann-pipnn
cargo build -p diskann-disk --features pipnn
```

- [ ] **Step 4: Review diff for accidental drift**

Check for and remove unrelated edits, especially formatting-only churn in `diskann-pipnn/src/partition.rs` or any new naming that reintroduces “sharded Vamana” semantics for PiPNN.

- [ ] **Step 5: Only after all targeted checks are green, consider commit/PR work**

No commit is part of this plan unless the user explicitly asks for it.

---

## Definition of Done

- [ ] One-shot PiPNN remains unchanged when the estimate fits the budget
- [ ] Over-budget PiPNN uses the corrected internal streaming RBC path
- [ ] No Vamana-style PiPNN data-shard build path was introduced
- [ ] Current PiPNN leaf build and HashPrune code are reused instead of reimplemented
- [ ] `final_prune=false` over-budget path is supported and verified
- [ ] Targeted tests and builds pass
- [ ] No unrelated diff noise remains
