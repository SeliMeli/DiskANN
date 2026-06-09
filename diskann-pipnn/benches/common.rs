/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared fixtures for the PiPNN phase benches.
//!
//! Provides:
//! - [`Load`] — the 4 canonical workloads (dim × scale)
//! - [`Dataset`] — mmap'd fp16 file with zero-copy `&[f16]` view
//! - [`pool`] — single rayon pool shared across all benches in a process
//! - [`leaves_for`] — in-process cached partition output per load
//! - [`candidates_for`] — disk-cached pre-prune candidate graph per load
//! - [`Load::selected`] — respects `PIPNN_BENCH_SCALE=1m|10m|all` env

#![allow(dead_code)] // benches each use a subset; suppressing per-bench warnings

use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use diskann_pipnn::hash_prune::HashPrune;
use diskann_pipnn::leaf_build;
use diskann_pipnn::partition::{Leaf, PartitionConfig};
use diskann_pipnn::PiPNNConfig;
use diskann_vector::distance::Metric;
use half::f16;
use memmap2::Mmap;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPool;

// ---------------------------------------------------------------------------
// Load types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Load {
    BigAnn1M,
    BigAnn10M,
    Enron1M,
    Enron10M,
}

impl Load {
    pub const ALL: &'static [Load] = &[
        Load::BigAnn1M,
        Load::BigAnn10M,
        Load::Enron1M,
        Load::Enron10M,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Load::BigAnn1M => "bigann_1m",
            Load::BigAnn10M => "bigann_10m",
            Load::Enron1M => "enron_1m",
            Load::Enron10M => "enron_10m",
        }
    }

    pub fn from_name(s: &str) -> Option<Load> {
        match s {
            "bigann_1m" => Some(Load::BigAnn1M),
            "bigann_10m" => Some(Load::BigAnn10M),
            "enron_1m" => Some(Load::Enron1M),
            "enron_10m" => Some(Load::Enron10M),
            _ => None,
        }
    }

    pub fn data_filename(&self) -> &'static str {
        match self {
            Load::BigAnn1M => "bigann_1m_fp16.bin",
            Load::BigAnn10M => "bigann_10m_fp16.bin",
            Load::Enron1M => "enron_fp16.bin",
            Load::Enron10M => "enron_10m_fp16.bin",
        }
    }

    pub fn npoints(&self) -> usize {
        match self {
            Load::BigAnn1M => 1_000_000,
            Load::BigAnn10M => 10_000_000,
            Load::Enron1M => 1_087_932,
            Load::Enron10M => 10_000_000,
        }
    }

    pub fn ndims(&self) -> usize {
        match self {
            Load::BigAnn1M | Load::BigAnn10M => 128,
            Load::Enron1M | Load::Enron10M => 384,
        }
    }

    pub fn metric(&self) -> Metric {
        match self {
            Load::BigAnn1M | Load::BigAnn10M => Metric::L2,
            Load::Enron1M | Load::Enron10M => Metric::CosineNormalized,
        }
    }

    pub fn is_10m(&self) -> bool {
        matches!(self, Load::BigAnn10M | Load::Enron10M)
    }

    /// PartitionConfig used when benching partition-only and as the upstream for
    /// leaf/final_prune fixtures.
    pub fn partition_config(&self) -> PartitionConfig {
        PartitionConfig {
            c_max: 256,
            c_min: 16,
            p_samp: 0.005,
            fanout: vec![8, 3],
            metric: self.metric(),
            leader_cap: 1000,
        }
    }

    /// Full PiPNNConfig for candidate-graph construction. `max_degree` and
    /// `alpha` are the knobs the final_prune bench sweeps on top of the cached
    /// fixture, but we set canonical defaults so one cached graph is enough.
    pub fn pipnn_config(&self) -> PiPNNConfig {
        let num_hash_planes = match self {
            Load::BigAnn1M | Load::BigAnn10M => 12,
            Load::Enron1M | Load::Enron10M => 14,
        };
        PiPNNConfig {
            num_hash_planes,
            c_max: 256,
            c_min: 16,
            p_samp: 0.005,
            fanout: vec![8, 3],
            k: 2,
            max_degree: 64,
            replicas: 1,
            l_max: 64,
            metric: self.metric(),
            final_prune: true,
            alpha: 1.2,
            num_threads: 16,
            leader_cap: 1000,
            saturate_after_prune: true,
        }
    }

    /// Yield loads filtered by the `PIPNN_BENCH_SCALE` env:
    /// - `1m` (default) → 1M loads only
    /// - `10m` → 10M loads only
    /// - `all` → all 4 loads
    pub fn selected() -> impl Iterator<Item = Load> {
        let scale = std::env::var("PIPNN_BENCH_SCALE").unwrap_or_else(|_| "1m".to_string());
        let filter: fn(&Load) -> bool = match scale.as_str() {
            "10m" => |l: &Load| l.is_10m(),
            "all" => |_: &Load| true,
            _ => |l: &Load| !l.is_10m(),
        };
        Load::ALL.iter().copied().filter(filter)
    }
}

// ---------------------------------------------------------------------------
// Dataset (mmap'd fp16, zero-copy view)
// ---------------------------------------------------------------------------

pub struct Dataset {
    _mmap: Mmap,
    data_start: usize,
    npoints: usize,
    ndims: usize,
}

impl Dataset {
    pub fn open(
        path: &Path,
        expected_npoints: usize,
        expected_ndims: usize,
    ) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: dataset files are owned by the bench harness and are not
        // modified concurrently while benches run. If a caller replaces the
        // file mid-run, behavior is UB, but that's outside the bench contract.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too short to contain DiskANN header",
            ));
        }
        let npoints = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let ndims = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        if npoints != expected_npoints || ndims != expected_ndims {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "header mismatch: file has {}x{}, expected {}x{}",
                    npoints, ndims, expected_npoints, expected_ndims
                ),
            ));
        }
        let expected_bytes = 8 + npoints * ndims * std::mem::size_of::<f16>();
        if mmap.len() < expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "file truncated: have {} bytes, need {}",
                    mmap.len(),
                    expected_bytes
                ),
            ));
        }
        Ok(Self {
            _mmap: mmap,
            data_start: 8,
            npoints,
            ndims,
        })
    }

    /// Zero-copy view of the data as `&[f16]`. Data is 2-byte aligned
    /// (mmap is page-aligned, header is 8 bytes).
    pub fn data(&self) -> &[f16] {
        let bytes = &self._mmap[self.data_start..self.data_start + self.npoints * self.ndims * 2];
        bytemuck::cast_slice(bytes)
    }

    pub fn npoints(&self) -> usize {
        self.npoints
    }

    pub fn ndims(&self) -> usize {
        self.ndims
    }
}

pub fn dataset_dir() -> PathBuf {
    match std::env::var("DISKANN_DATASETS") {
        Ok(s) => PathBuf::from(s),
        Err(_) => PathBuf::from("datasets"),
    }
}

/// Returns `None` and logs a `[skip]` message if the file doesn't exist or
/// can't be opened. This lets benches run partial matrices without crashing.
pub fn dataset_for(load: Load) -> Option<Arc<Dataset>> {
    type Cache = Mutex<HashMap<Load, Arc<Dataset>>>;
    static CACHE: OnceCell<Cache> = OnceCell::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock();
    if let Some(ds) = cache.get(&load) {
        return Some(ds.clone());
    }
    let path = dataset_dir().join(load.data_filename());
    if !path.exists() {
        eprintln!(
            "[skip] {}: dataset not found at {} (set $DISKANN_DATASETS or place file under ./datasets/)",
            load.name(),
            path.display()
        );
        return None;
    }
    match Dataset::open(&path, load.npoints(), load.ndims()) {
        Ok(ds) => {
            let arc = Arc::new(ds);
            cache.insert(load, arc.clone());
            Some(arc)
        }
        Err(e) => {
            eprintln!("[skip] {}: {} ({})", load.name(), path.display(), e);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Rayon pool
// ---------------------------------------------------------------------------

/// Shared 16-thread pool. Every bench runs its phase call inside
/// `pool().install(|| ...)`.
pub fn pool() -> &'static ThreadPool {
    static POOL: OnceCell<ThreadPool> = OnceCell::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .expect("failed to build rayon pool")
    })
}

// ---------------------------------------------------------------------------
// Leaves fixture (in-process, per process)
// ---------------------------------------------------------------------------

pub fn leaves_for(load: Load) -> Option<Arc<Vec<Leaf>>> {
    type Cache = Mutex<HashMap<Load, Arc<Vec<Leaf>>>>;
    static CACHE: OnceCell<Cache> = OnceCell::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock();
        if let Some(arc) = guard.get(&load) {
            return Some(arc.clone());
        }
    }
    let dataset = dataset_for(load)?;
    let cfg = load.partition_config();
    eprintln!(
        "[{}] building leaves (c_max={} fanout={:?})...",
        load.name(),
        cfg.c_max,
        cfg.fanout
    );
    let t0 = Instant::now();
    let leaves = pool().install(|| {
        diskann_pipnn::partition::partition(dataset.data(), load.ndims(), load.npoints(), &cfg, 42)
    });
    eprintln!(
        "[{}] leaves: {} built in {:.2}s",
        load.name(),
        leaves.len(),
        t0.elapsed().as_secs_f64()
    );
    let arc = Arc::new(leaves);
    cache.lock().insert(load, arc.clone());
    Some(arc)
}

// ---------------------------------------------------------------------------
// Candidate-graph fixture (disk-cached)
// ---------------------------------------------------------------------------

pub type CandidateGraph = Vec<Vec<(u32, f32)>>;

pub fn candidates_for(load: Load) -> Option<Arc<CandidateGraph>> {
    type Cache = Mutex<HashMap<Load, Arc<CandidateGraph>>>;
    static CACHE: OnceCell<Cache> = OnceCell::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock();
        if let Some(arc) = guard.get(&load) {
            return Some(arc.clone());
        }
    }

    let dataset = dataset_for(load)?;
    let data_path = dataset_dir().join(load.data_filename());
    let cfg = load.pipnn_config();
    let meta = CacheMeta::from_load(load, &data_path, &cfg).ok()?;

    // Disk cache path
    let cache_path = bench_fixtures_dir().join(format!(
        "{}-{:016x}.candidates",
        load.name(),
        meta.config_hash
    ));

    // Try load
    if cache_path.exists() {
        match load_candidates(&cache_path, &meta) {
            Ok(Some(graph)) => {
                eprintln!(
                    "[{}] loaded candidates from cache ({} nodes, {:.2} GB, {})",
                    load.name(),
                    graph.len(),
                    cache_path.metadata().map(|m| m.len() as f64).unwrap_or(0.0) / 1e9,
                    cache_path.display()
                );
                let arc = Arc::new(graph);
                cache.lock().insert(load, arc.clone());
                return Some(arc);
            }
            Ok(None) => {
                eprintln!("[{}] cache metadata mismatch; rebuilding", load.name());
            }
            Err(e) => {
                eprintln!("[{}] cache read failed ({}); rebuilding", load.name(), e);
            }
        }
    }

    // Rebuild the pre-prune graph: partition + leaf_build + hash_prune extract
    eprintln!("[{}] building candidate graph...", load.name());
    let t0 = Instant::now();

    let leaves = leaves_for(load)?;
    let data = dataset.data();
    let ndims = load.ndims();
    let npoints = load.npoints();

    // Allow: the `par_iter().for_each` runs inside `pool.install`, so rayon
    // work stays on the bench pool rather than leaking to the global one.
    #[allow(clippy::disallowed_methods)]
    let graph = pool().install(|| {
        let hash_prune = HashPrune::new(
            data,
            npoints,
            ndims,
            cfg.num_hash_planes,
            cfg.l_max,
            cfg.max_degree,
            42,
        );
        leaves.par_iter().for_each(|leaf| {
            let indices_usize: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let edges = leaf_build::build_leaf(data, ndims, &indices_usize, cfg.k, cfg.metric);
            hash_prune.add_edges_batched(&edges);
        });
        hash_prune.extract_graph_for_prune()
    });

    eprintln!(
        "[{}] candidate graph: {} nodes in {:.2}s",
        load.name(),
        graph.len(),
        t0.elapsed().as_secs_f64()
    );

    // Save to disk (best effort — don't fail the bench if save fails)
    if let Err(e) = std::fs::create_dir_all(bench_fixtures_dir()) {
        eprintln!("[{}] cache dir create failed: {}", load.name(), e);
    } else if let Err(e) = save_candidates(&cache_path, &graph, &meta) {
        eprintln!("[{}] cache write failed: {}", load.name(), e);
    } else {
        eprintln!("[{}] cached to {}", load.name(), cache_path.display());
    }

    let arc = Arc::new(graph);
    cache.lock().insert(load, arc.clone());
    Some(arc)
}

// ---------------------------------------------------------------------------
// Cache location and serialization
// ---------------------------------------------------------------------------

fn bench_fixtures_dir() -> PathBuf {
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target).join("bench-fixtures")
}

const CACHE_MAGIC: &[u8; 4] = b"PPCG";
const CACHE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct CacheMeta {
    config_hash: u64,
    data_path_hash: u64,
    data_mtime_secs: u64,
    data_mtime_nanos: u32,
    npoints: u64,
    ndims: u32,
}

impl CacheMeta {
    fn from_load(load: Load, data_path: &Path, cfg: &PiPNNConfig) -> std::io::Result<Self> {
        let meta = fs::metadata(data_path)?;
        let mtime = meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Canonicalize path for stable hashing across working-directory changes.
        let canon = fs::canonicalize(data_path).unwrap_or_else(|_| data_path.to_path_buf());
        canon.hash(&mut hasher);
        let data_path_hash = hasher.finish();

        let mut cfg_hasher = std::collections::hash_map::DefaultHasher::new();
        cfg.c_max.hash(&mut cfg_hasher);
        cfg.c_min.hash(&mut cfg_hasher);
        cfg.p_samp.to_bits().hash(&mut cfg_hasher);
        cfg.fanout.hash(&mut cfg_hasher);
        cfg.k.hash(&mut cfg_hasher);
        cfg.l_max.hash(&mut cfg_hasher);
        cfg.num_hash_planes.hash(&mut cfg_hasher);
        cfg.metric.as_str().hash(&mut cfg_hasher);
        load.name().hash(&mut cfg_hasher);
        let config_hash = cfg_hasher.finish();

        Ok(Self {
            config_hash,
            data_path_hash,
            data_mtime_secs: mtime.as_secs(),
            data_mtime_nanos: mtime.subsec_nanos(),
            npoints: load.npoints() as u64,
            ndims: load.ndims() as u32,
        })
    }
}

fn save_candidates(path: &Path, graph: &CandidateGraph, meta: &CacheMeta) -> std::io::Result<()> {
    let tmp = path.with_extension("candidates.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);

        // Header
        w.write_all(CACHE_MAGIC)?;
        w.write_all(&CACHE_VERSION.to_le_bytes())?;
        w.write_all(&meta.config_hash.to_le_bytes())?;
        w.write_all(&meta.data_path_hash.to_le_bytes())?;
        w.write_all(&meta.data_mtime_secs.to_le_bytes())?;
        w.write_all(&meta.data_mtime_nanos.to_le_bytes())?;
        w.write_all(&meta.npoints.to_le_bytes())?;
        w.write_all(&meta.ndims.to_le_bytes())?;
        // 4-byte pad so entries/offsets start 8-byte aligned in the file.
        w.write_all(&0u32.to_le_bytes())?;

        // Offsets (prefix-sum of entry counts). npoints+1 u64s.
        let npoints = meta.npoints as usize;
        let mut cum: u64 = 0;
        w.write_all(&cum.to_le_bytes())?;
        for inner in graph.iter().take(npoints) {
            cum += inner.len() as u64;
            w.write_all(&cum.to_le_bytes())?;
        }

        // Entries: [node: u32, dist_bits: u32] per entry
        for inner in graph.iter().take(npoints) {
            for &(node, dist) in inner {
                w.write_all(&node.to_le_bytes())?;
                w.write_all(&dist.to_bits().to_le_bytes())?;
            }
        }

        w.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Returns `Ok(None)` on metadata mismatch (not an error — caller rebuilds).
fn load_candidates(path: &Path, expected: &CacheMeta) -> std::io::Result<Option<CandidateGraph>> {
    let mut r = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        return Ok(None);
    }
    let version = read_u32(&mut r)?;
    if version != CACHE_VERSION {
        return Ok(None);
    }
    let config_hash = read_u64(&mut r)?;
    let data_path_hash = read_u64(&mut r)?;
    let data_mtime_secs = read_u64(&mut r)?;
    let data_mtime_nanos = read_u32(&mut r)?;
    let npoints = read_u64(&mut r)?;
    let ndims = read_u32(&mut r)?;
    let _pad = read_u32(&mut r)?;

    if config_hash != expected.config_hash
        || data_path_hash != expected.data_path_hash
        || data_mtime_secs != expected.data_mtime_secs
        || data_mtime_nanos != expected.data_mtime_nanos
        || npoints != expected.npoints
        || ndims != expected.ndims
    {
        return Ok(None);
    }

    let npoints = npoints as usize;

    // Offsets
    let mut offsets: Vec<u64> = Vec::with_capacity(npoints + 1);
    let mut buf = [0u8; 8];
    for _ in 0..(npoints + 1) {
        r.read_exact(&mut buf)?;
        offsets.push(u64::from_le_bytes(buf));
    }

    // Entries
    let total_entries = offsets[npoints] as usize;
    let mut graph: CandidateGraph = Vec::with_capacity(npoints);
    let remaining_per_node: Vec<usize> = (0..npoints)
        .map(|i| (offsets[i + 1] - offsets[i]) as usize)
        .collect();

    // Read entries node by node using remaining_per_node; one big read would be
    // faster but we'd need separate alloc. Inner Vec allocated exactly.
    let mut entry_buf = [0u8; 8];
    for &count in remaining_per_node.iter().take(npoints) {
        let mut inner = Vec::with_capacity(count);
        for _ in 0..count {
            r.read_exact(&mut entry_buf)?;
            let node = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
            let dist_bits = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());
            inner.push((node, f32::from_bits(dist_bits)));
        }
        graph.push(inner);
    }

    debug_assert_eq!(graph.iter().map(|v| v.len()).sum::<usize>(), total_entries);
    let _ = remaining_per_node; // kept only for assertion readability

    Ok(Some(graph))
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
