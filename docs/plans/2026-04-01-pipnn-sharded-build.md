# PiPNN Sharded Build Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable PiPNN to build graphs on datasets larger than available RAM by processing data in shards while keeping HashPrune global.

**Architecture:** Data is processed shard-by-shard (via mmap windows), while LSH sketches and HashPrune reservoirs remain global in memory. The decision between one-shot and sharded build is automatic based on RAM budget, matching Vamana's `determine_build_strategy` pattern. No merge step needed — global HashPrune produces the final unified graph.

**Tech Stack:** Rust, rayon, memmap2, diskann-pipnn, diskann-disk

**Branch:** `pipnn-sharded` (worktree at `/home/weiyaoluo/Diskann/.claude/worktrees/pipnn-sharded/`)

---

## Memory Model

```
One-shot (current):  data(N×d×2) + HashPrune(N×l_max×8) + LSH(N×planes×4) + buffers
Sharded:             shard(N/K×d×2) + HashPrune(N×l_max×8) + LSH(N×planes×4) + buffers
                     ↑ only this shrinks
```

RAM budget formula:
```
fixed = N × l_max × 8  +  N × num_planes × 4  +  500MB overhead
shard_data = shard_npoints × ndims × sizeof(T)
total = fixed + shard_data
shard_npoints = (ram_budget - fixed) / (ndims × sizeof(T))
nshards = ceil(N / shard_npoints)
if nshards == 1: use one-shot path (no overhead)
```

## File Structure

### New files
- `diskann-pipnn/src/data_source.rs` — `VectorDataSource` trait + `SliceDataSource` + `MmapDataSource`

### Modified files
- `diskann-pipnn/src/lib.rs` — add `pub mod data_source`
- `diskann-pipnn/src/hash_prune.rs` — add `LshSketchesBuilder` for incremental sketch computation
- `diskann-pipnn/src/partition.rs` — change `data: &[T]` to `data: &(impl VectorDataSource<Elem=T> + ?Sized)` in 4 functions
- `diskann-pipnn/src/leaf_build.rs` — change `data: &[T]` to `data: &(impl VectorDataSource<Elem=T> + ?Sized)` in 2 functions
- `diskann-pipnn/src/builder.rs` — add `build_sharded()`, wrap existing paths with `SliceDataSource`
- `diskann-disk/src/build/builder/build.rs` — add `estimate_pipnn_ram`, route to sharded build
- `diskann-pipnn/Cargo.toml` — add `memmap2` dependency

---

## Task 1: VectorDataSource trait + SliceDataSource

**Files:**
- Create: `diskann-pipnn/src/data_source.rs`
- Modify: `diskann-pipnn/src/lib.rs`

- [ ] **Step 1: Write the trait and SliceDataSource**

```rust
// diskann-pipnn/src/data_source.rs
use diskann::utils::VectorRepr;

/// Abstraction over vector data — in-memory slice or memory-mapped file.
pub trait VectorDataSource: Send + Sync {
    type Elem: VectorRepr + Send + Sync;
    fn get(&self, idx: usize) -> &[Self::Elem];
    fn npoints(&self) -> usize;
    fn ndims(&self) -> usize;
}

/// Zero-cost wrapper around a contiguous &[T] slice.
pub struct SliceDataSource<'a, T> {
    data: &'a [T],
    npoints: usize,
    ndims: usize,
}

impl<'a, T: VectorRepr + Send + Sync> SliceDataSource<'a, T> {
    pub fn new(data: &'a [T], npoints: usize, ndims: usize) -> Self {
        debug_assert_eq!(data.len(), npoints * ndims);
        Self { data, npoints, ndims }
    }
}

impl<'a, T: VectorRepr + Send + Sync> VectorDataSource for SliceDataSource<'a, T> {
    type Elem = T;
    #[inline(always)]
    fn get(&self, idx: usize) -> &[T] {
        &self.data[idx * self.ndims..(idx + 1) * self.ndims]
    }
    fn npoints(&self) -> usize { self.npoints }
    fn ndims(&self) -> usize { self.ndims }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod data_source;` to `diskann-pipnn/src/lib.rs`.

- [ ] **Step 3: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_slice_data_source() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src = SliceDataSource::new(&data, 2, 3);
        assert_eq!(src.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(src.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(src.npoints(), 2);
        assert_eq!(src.ndims(), 3);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p diskann-pipnn --lib data_source`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diskann-pipnn/src/data_source.rs diskann-pipnn/src/lib.rs
git commit -m "feat: add VectorDataSource trait + SliceDataSource"
```

---

## Task 2: MmapDataSource

**Files:**
- Modify: `diskann-pipnn/src/data_source.rs`
- Modify: `diskann-pipnn/Cargo.toml`

- [ ] **Step 1: Add memmap2 dependency**

Add `memmap2 = "0.9"` to `[dependencies]` in `diskann-pipnn/Cargo.toml`.

- [ ] **Step 2: Implement MmapDataSource**

```rust
use std::sync::Arc;

/// Memory-mapped vector data from a DiskANN .bin file.
/// Header: 4 bytes npoints (u32) + 4 bytes ndims (u32), then data.
/// OS pages in/out on demand — only active pages consume RSS.
pub struct MmapDataSource<T: VectorRepr + Send + Sync> {
    mmap: Arc<memmap2::Mmap>,
    npoints: usize,
    ndims: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: VectorRepr + Send + Sync + bytemuck::Pod> MmapDataSource<T> {
    /// Open a DiskANN .bin file and mmap it. Optionally restrict to a
    /// window of [offset..offset+count) points for sharded access.
    pub fn open(path: &str, offset: usize, count: Option<usize>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });

        let header = &mmap[..8];
        let total_npoints = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let ndims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let npoints = count.unwrap_or(total_npoints - offset).min(total_npoints - offset);

        Ok(Self { mmap, npoints, ndims, _phantom: std::marker::PhantomData })
    }
}
```

The `get()` implementation computes the byte offset from the header + point index.

- [ ] **Step 3: Write test**

Test with a temp file written in DiskANN format, then mmap'd and read back.

- [ ] **Step 4: Run tests, commit**

---

## Task 3: LshSketchesBuilder for incremental computation

**Files:**
- Modify: `diskann-pipnn/src/hash_prune.rs`

- [ ] **Step 1: Add LshSketchesBuilder**

```rust
/// Incremental builder for LSH sketches — compute shard by shard.
pub struct LshSketchesBuilder {
    hyperplanes: Vec<f32>,
    sketches: Vec<f32>,
    num_planes: usize,
    ndims: usize,
    npoints: usize,
}

impl LshSketchesBuilder {
    pub fn new(npoints: usize, ndims: usize, num_planes: usize, seed: u64) -> Self {
        // Generate hyperplanes (same seed = same planes as LshSketches::new)
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<f32> = (0..num_planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let sketches = vec![0.0f32; npoints * num_planes];
        Self { hyperplanes, sketches, num_planes, ndims, npoints }
    }

    /// Compute sketches for points [global_offset..global_offset+shard_npoints).
    /// Can be called multiple times for different shards (non-overlapping ranges).
    pub fn fill_shard<T: VectorRepr + Send + Sync>(
        &mut self, data: &[T], global_offset: usize, shard_npoints: usize,
    ) { /* same dot-product loop as LshSketches::new, writing to self.sketches[offset..] */ }

    /// Consume builder into LshSketches.
    pub fn finish(self) -> LshSketches {
        LshSketches { num_planes: self.num_planes, sketches: self.sketches, npoints: self.npoints }
    }
}
```

- [ ] **Step 2: Refactor LshSketches::new to use builder internally**

```rust
impl LshSketches {
    pub fn new<T: VectorRepr + Send + Sync>(...) -> Self {
        let mut builder = LshSketchesBuilder::new(npoints, ndims, num_planes, seed);
        builder.fill_shard(data, 0, npoints);
        builder.finish()
    }
}
```

- [ ] **Step 3: Add fill_shard_quantized for SQ path**

Same structure but uses bit-scanning dot product from `new_from_quantized`.

- [ ] **Step 4: Test that builder produces identical sketches to current new()**

- [ ] **Step 5: Commit**

---

## Task 4: Migrate partition.rs to VectorDataSource

**Files:**
- Modify: `diskann-pipnn/src/partition.rs`

- [ ] **Step 1: Change partition_assign signature**

```rust
// Before: fn partition_assign<T: VectorRepr + Send + Sync>(data: &[T], ndims: usize, ...)
// After:  fn partition_assign<T: VectorRepr + Send + Sync>(data: &(impl VectorDataSource<Elem=T> + ?Sized), ...)
```

Replace `&data[idx * ndims..(idx + 1) * ndims]` with `data.get(idx)`. Remove `ndims` parameter (get from `data.ndims()`).

- [ ] **Step 2: Propagate to partition, parallel_partition, merge_small_into_nearest**

Same pattern: `&[T]` → `&(impl VectorDataSource<Elem=T> + ?Sized)`.

- [ ] **Step 3: Update builder.rs callers to wrap with SliceDataSource**

In `build_internal_impl`, before calling partition:
```rust
let data_source = SliceDataSource::new(data, npoints, ndims);
let leaves = parallel_partition(&data_source, &indices, &partition_config, seed);
```

- [ ] **Step 4: Run all tests — must be identical**

Run: `cargo test -p diskann-pipnn --lib`
Expected: 128 tests pass, no regression

- [ ] **Step 5: Commit**

---

## Task 5: Migrate leaf_build.rs to VectorDataSource

**Files:**
- Modify: `diskann-pipnn/src/leaf_build.rs`

- [ ] **Step 1: Change build_leaf and build_leaf_with_buffers signatures**

Same pattern as Task 4. Replace `data[idx * ndims..(idx + 1) * ndims]` with `data.get(idx)`.

- [ ] **Step 2: Update builder.rs callers**

```rust
let edges = leaf_build::build_leaf(&data_source, &leaf.indices, config.k, config.metric);
```

- [ ] **Step 3: Run all tests**

- [ ] **Step 4: Commit**

---

## Task 6: Implement build_sharded in builder.rs

**Files:**
- Modify: `diskann-pipnn/src/builder.rs`

- [ ] **Step 1: Add build_sharded function**

```rust
/// Sharded PiPNN build for datasets larger than RAM.
///
/// Phase 0: Stream data shard-by-shard to compute LSH sketches + medoid
/// Phase 1: Global partition per shard → leaf build → edges to global HashPrune
/// Phase 2: Extract graph from HashPrune
///
/// Only one shard of data is in memory at any time.
pub fn build_sharded<S: VectorDataSource>(
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    num_shards: usize,
    shard_factory: impl Fn(usize) -> S,  // shard_id → data source
    shard_ranges: &[(usize, usize)],      // (global_offset, shard_npoints) per shard
) -> PiPNNResult<PiPNNGraph>
```

- [ ] **Step 2: Implement Phase 0 — streaming sketches + medoid**

```rust
let mut builder = LshSketchesBuilder::new(npoints, ndims, config.num_hash_planes, 42);
let mut centroid = vec![0.0f32; ndims];
for (shard_id, &(offset, shard_n)) in shard_ranges.iter().enumerate() {
    let shard = shard_factory(shard_id);
    builder.fill_shard_from_source(&shard, offset, shard_n);
    // accumulate centroid
    drop(shard);
}
let sketches = builder.finish();
let medoid = find_medoid_from_centroid(...); // second pass
```

- [ ] **Step 3: Implement Phase 1 — per-shard partition + leaf build**

```rust
let hash_prune = HashPrune::from_sketches(sketches, npoints, config.l_max, config.max_degree);
for replica in 0..config.replicas {
    for (shard_id, &(offset, shard_n)) in shard_ranges.iter().enumerate() {
        let shard = shard_factory(shard_id);
        let indices: Vec<usize> = (0..shard_n).collect();
        let leaves = parallel_partition(&shard, &indices, &config, seed);
        leaves.par_iter().for_each(|leaf| {
            let edges = build_leaf(&shard, &leaf.indices, config.k, config.metric);
            // Translate shard-local indices to global
            let global_edges: Vec<Edge> = edges.iter().map(|e| Edge {
                src: e.src + offset, dst: e.dst + offset, distance: e.distance,
            }).collect();
            hash_prune.add_edges_batched(&global_edges);
        });
        drop(shard);
    }
}
```

- [ ] **Step 4: Phase 2 — extract graph**

- [ ] **Step 5: Test with synthetic 3-shard dataset**

Create 3000 random points, build with num_shards=3 and num_shards=1, verify recall is within 1%.

- [ ] **Step 6: Commit**

---

## Task 7: Wire into diskann-disk build pipeline

**Files:**
- Modify: `diskann-disk/src/build/builder/build.rs`

- [ ] **Step 1: Add estimate_pipnn_ram_usage**

```rust
fn estimate_pipnn_ram_usage(npoints: usize, ndims: usize, type_size: usize, l_max: usize, num_planes: usize) -> f64 {
    let data = npoints as f64 * ndims as f64 * type_size as f64;
    let reservoirs = npoints as f64 * l_max as f64 * 8.0;
    let sketches = npoints as f64 * num_planes as f64 * 4.0;
    let overhead = 500.0 * 1024.0 * 1024.0; // 500 MB
    data + reservoirs + sketches + overhead
}
```

- [ ] **Step 2: Add routing logic in build_sync_pipnn**

```rust
fn build_sync_pipnn(&mut self) -> ANNResult<()> {
    let estimated = estimate_pipnn_ram_usage(...);
    let ram_limit = self.disk_build_param.build_memory_limit().in_bytes() as f64;
    if estimated > ram_limit {
        return self.build_sync_pipnn_sharded();
    }
    // existing one-shot path...
}
```

- [ ] **Step 3: Implement build_sync_pipnn_sharded**

Uses `MmapDataSource` windows into the dataset file, calls `builder::build_sharded`.

- [ ] **Step 4: Test one-shot path unchanged (Enron 1M with high RAM limit)**

- [ ] **Step 5: Test sharded path triggers (Enron 1M with low RAM limit, e.g., 1 GB)**

- [ ] **Step 6: Commit**

---

## Task 8: Integration testing on VM

- [ ] **Step 1: Run Enron 1M PiPNN FP one-shot — verify no regression**

Same config as current quality gate: c_max=256, c_min=16, etc.
Expected: ~18s total, 96.65% recall.

- [ ] **Step 2: Run Enron 1M PiPNN FP sharded (force 2 shards via low RAM limit)**

Set `build_ram_limit_gb: 1.5` to force sharding.
Compare recall and build time vs one-shot.

- [ ] **Step 3: Run Enron 1M Vamana merged vs PiPNN sharded**

Both with same low RAM limit. Compare build time.

- [ ] **Step 4: Commit final, push**

---

## Constraints

- `final_prune=true` is incompatible with sharded builds (requires full data). Return error at config validation.
- Sharded build produces identical results to one-shot when num_shards=1.
- The SQ path already handles data-not-in-memory via `chunked_quantize_and_medoid` — the sharded build extends this pattern to the FP path.
- No intermediate files for edges or graphs. HashPrune is the single accumulator.
- Each shard's partition is intra-shard only. Cross-shard connectivity comes from the fanout overlap within each shard's RBC, plus multiple replicas if configured.
