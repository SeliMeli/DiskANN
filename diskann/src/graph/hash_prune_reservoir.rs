/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! 1:1 copy of PiPNN's HashPrune reservoir (`diskann-pipnn/src/hash_prune.rs`),
//! lifted into the `diskann` crate so Vamana can absorb edges **online** into a
//! bounded per-node reservoir — the way HashPrune actually works (LSH bucket +
//! find_hash + farthest-eviction, no occlusion, no re-prune).
//!
//! `OnlineHashPrune` is PiPNN's persistent global structure: one `HotSlot` per
//! point (16 B, with a `parking_lot::RawMutex` for concurrent `add_edge`) over
//! three contiguous AoSoA cold slabs (hashes / bf16-distances / neighbors). Each
//! `add_edge(p, c, hash, dist)` locks slot `p` and runs `insert_locked` (PiPNN
//! verbatim); `get_neighbors(p)` drains it sorted. The hash is supplied by the
//! caller (computed from the existing `LshState`), so only the reservoir is
//! copied, not the sketch machinery.
#![allow(unsafe_op_in_unsafe_fn)]

use parking_lot::lock_api::RawMutex as RawMutexTrait;

/// Matches PiPNN `L_MAX_MAX` (hard-coded 128 to match `PiPNNConfig` default).
pub const L_MAX_MAX: usize = 128;

/// Truncate `f32` to bf16 (PiPNN `diskann_vector::bf16`), order-preserving for
/// the non-negative distances stored here.
#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Reconstruct `f32` from a bf16, zero-filling the lower mantissa bits.
#[inline(always)]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ─── Inlined runtime SIMD dispatch (PiPNN cpu_dispatch.rs, tracing removed) ────
mod cpu_dispatch {
    use std::sync::OnceLock;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(crate) enum SimdTier {
        Avx512,
        Avx2,
        Scalar,
    }

    static TIER: OnceLock<SimdTier> = OnceLock::new();

    #[inline]
    pub(crate) fn tier() -> SimdTier {
        *TIER.get_or_init(detect_tier)
    }

    fn detect_tier() -> SimdTier {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return SimdTier::Avx512;
            }
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
            {
                return SimdTier::Avx2;
            }
        }
        SimdTier::Scalar
    }
}

// ─── HotSlot: 16-byte per-point mutex + cached fields (PiPNN verbatim) ─────────

#[repr(C, align(16))]
struct HotSlot {
    lock: parking_lot::RawMutex,
    len: u8,
    farthest_idx: u8,
    _pad0: u8,
    farthest_dist: u16,
    _pad1: [u8; 10],
}

impl HotSlot {
    const fn new_empty() -> Self {
        Self {
            lock: <parking_lot::RawMutex as RawMutexTrait>::INIT,
            len: 0,
            farthest_idx: 0,
            _pad0: 0,
            farthest_dist: 0,
            _pad1: [0; 10],
        }
    }
}

const _: () = assert!(std::mem::size_of::<HotSlot>() == 16);

// ─── Cold slabs view ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct ColdSlotPtrs {
    hashes: *mut u16,
    distances: *mut u16,
    neighbors: *mut u32,
    scan_lanes: usize,
}

#[inline(always)]
fn round_up_to_32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

// ─── find_hash SIMD: 32-way u16 compare (PiPNN verbatim) ──────────────────────

/// SAFETY: `hashes` must point at `scan_lanes` valid `u16` slots. `len` must
/// be the number of meaningful entries (<= scan_lanes and <= 128).
#[inline(always)]
unsafe fn find_hash_simd(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        match cpu_dispatch::tier() {
            cpu_dispatch::SimdTier::Avx512 => {
                return find_hash_avx512(hashes, scan_lanes, len, target);
            }
            cpu_dispatch::SimdTier::Avx2 => {
                return find_hash_avx2(hashes, scan_lanes, len, target);
            }
            cpu_dispatch::SimdTier::Scalar => {}
        }
    }
    find_hash_scalar(hashes, len as usize, target)
}

/// SAFETY: caller guarantees AVX-512F + AVX-512BW at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn find_hash_avx512(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let len = len as usize;
    let t = _mm512_set1_epi16(target as i16);
    let chunks = scan_lanes / 32;
    let mut combined: u128 = 0;
    for chunk in 0..chunks {
        let v = _mm512_loadu_si512(hashes.add(chunk * 32) as *const __m512i);
        let m = _mm512_cmpeq_epi16_mask(v, t) as u128;
        combined |= m << (chunk * 32);
    }
    let len_mask: u128 = if len >= 128 { u128::MAX } else { (1u128 << len) - 1 };
    let valid = combined & len_mask;
    if valid != 0 {
        Some(valid.trailing_zeros() as usize)
    } else {
        None
    }
}

/// SAFETY: caller guarantees AVX2 at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_hash_avx2(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let len = len as usize;
    let t = _mm256_set1_epi16(target as i16);
    let chunks = scan_lanes / 16;
    for chunk in 0..chunks {
        let v = _mm256_loadu_si256(hashes.add(chunk * 16) as *const __m256i);
        let m = _mm256_cmpeq_epi16(v, t);
        let bits = _mm256_movemask_epi8(m) as u32;
        if bits != 0 {
            let lane = chunk * 16 + (bits.trailing_zeros() as usize) / 2;
            if lane < len {
                return Some(lane);
            }
        }
    }
    None
}

/// SAFETY: `hashes` must point at `len` valid `u16` slots.
#[inline(always)]
unsafe fn find_hash_scalar(hashes: *const u16, len: usize, target: u16) -> Option<usize> {
    for i in 0..len {
        if *hashes.add(i) == target {
            return Some(i);
        }
    }
    None
}

// ─── Per-reservoir mutation helpers (PiPNN verbatim) ──────────────────────────

/// SAFETY: pointers in `cold` are valid for `scan_lanes` elements each.
#[inline]
unsafe fn update_farthest(hot: &mut HotSlot, cold: ColdSlotPtrs) {
    if hot.len == 0 {
        hot.farthest_dist = 0;
        hot.farthest_idx = 0;
        return;
    }
    let mut max_dist: u16 = 0;
    let mut max_idx: u8 = 0;
    for i in 0..hot.len as usize {
        let d = *cold.distances.add(i);
        if d > max_dist {
            max_dist = d;
            max_idx = i as u8;
        }
    }
    hot.farthest_dist = max_dist;
    hot.farthest_idx = max_idx;
}

/// SAFETY: pointers in `cold` are valid for `scan_lanes` elements each, and
/// `l_max <= scan_lanes`.
#[inline(always)]
unsafe fn insert_locked(
    hot: &mut HotSlot,
    cold: ColdSlotPtrs,
    hash: u16,
    neighbor: u32,
    distance: f32,
    l_max: u8,
) -> bool {
    let dist_bf16 = f32_to_bf16(distance);

    if hot.len >= l_max && dist_bf16 >= hot.farthest_dist {
        return false;
    }

    if let Some(idx) = find_hash_simd(cold.hashes, cold.scan_lanes, hot.len, hash) {
        if dist_bf16 < *cold.distances.add(idx) {
            let was_farthest = idx == hot.farthest_idx as usize;
            *cold.neighbors.add(idx) = neighbor;
            *cold.distances.add(idx) = dist_bf16;
            if was_farthest {
                update_farthest(hot, cold);
            }
            return true;
        }
        return false;
    }

    if hot.len < l_max {
        let new_idx = hot.len as usize;
        *cold.hashes.add(new_idx) = hash;
        *cold.distances.add(new_idx) = dist_bf16;
        *cold.neighbors.add(new_idx) = neighbor;
        hot.len += 1;
        if dist_bf16 >= hot.farthest_dist {
            hot.farthest_dist = dist_bf16;
            hot.farthest_idx = new_idx as u8;
        }
        return true;
    }

    if dist_bf16 < hot.farthest_dist {
        let idx = hot.farthest_idx as usize;
        *cold.hashes.add(idx) = hash;
        *cold.distances.add(idx) = dist_bf16;
        *cold.neighbors.add(idx) = neighbor;
        update_farthest(hot, cold);
        return true;
    }
    false
}

/// SAFETY: pointers in `cold` are valid for `scan_lanes` elements each.
unsafe fn get_neighbors_saturated(
    hot: &HotSlot,
    cold: ColdSlotPtrs,
    max_degree: usize,
) -> Vec<(u32, f32)> {
    let n = hot.len as usize;
    let mut tmp: Vec<(u32, u16)> = (0..n)
        .map(|i| (*cold.neighbors.add(i), *cold.distances.add(i)))
        .collect();
    tmp.sort_unstable_by_key(|&(_, d)| d);
    tmp.truncate(max_degree);
    tmp.into_iter().map(|(id, d)| (id, bf16_to_f32(d))).collect()
}

// ─── OnlineHashPrune: PiPNN's persistent global reservoir (the online API) ────

/// One persistent HashPrune reservoir per point, over contiguous AoSoA slabs.
/// `add_edge` is the online insert (PiPNN's exact find_hash + farthest-eviction);
/// `get_neighbors` drains a node sorted. Disjoint-index parallel `add_edge` is
/// safe — each slot is guarded by its `HotSlot` lock.
pub struct OnlineHashPrune {
    hot: Vec<HotSlot>,
    cold_hashes: Vec<u16>,
    cold_distances: Vec<u16>,
    cold_neighbors: Vec<u32>,
    scan_lanes: usize,
    l_max: usize,
}

// SAFETY: HotSlot has interior mutability via RawMutex; cold slabs are plain
// bit-pattern arrays, each per-point slot guarded by HotSlot[i].lock.
unsafe impl Send for OnlineHashPrune {}
unsafe impl Sync for OnlineHashPrune {}

impl std::fmt::Debug for OnlineHashPrune {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineHashPrune")
            .field("npoints", &self.hot.len())
            .field("l_max", &self.l_max)
            .field("scan_lanes", &self.scan_lanes)
            .finish()
    }
}

impl OnlineHashPrune {
    pub fn new(npoints: usize, l_max: usize) -> Self {
        assert!(l_max <= L_MAX_MAX, "l_max {l_max} exceeds L_MAX_MAX {L_MAX_MAX}");
        let scan_lanes = round_up_to_32(l_max).max(32);
        let mut hot: Vec<HotSlot> = Vec::with_capacity(npoints);
        for _ in 0..npoints {
            hot.push(HotSlot::new_empty());
        }
        let total = npoints
            .checked_mul(scan_lanes)
            .expect("OnlineHashPrune slab size overflowed usize");
        Self {
            hot,
            cold_hashes: vec![0u16; total],
            cold_distances: vec![0u16; total],
            cold_neighbors: vec![0u32; total],
            scan_lanes,
            l_max,
        }
    }

    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// SAFETY: idx < self.hot.len().
    #[inline]
    unsafe fn slot_ptrs(&self, idx: usize) -> (*mut HotSlot, ColdSlotPtrs) {
        let hot_ptr = (self.hot.as_ptr() as *mut HotSlot).add(idx);
        let off = idx * self.scan_lanes;
        let cold = ColdSlotPtrs {
            hashes: (self.cold_hashes.as_ptr() as *mut u16).add(off),
            distances: (self.cold_distances.as_ptr() as *mut u16).add(off),
            neighbors: (self.cold_neighbors.as_ptr() as *mut u32).add(off),
            scan_lanes: self.scan_lanes,
        };
        (hot_ptr, cold)
    }

    #[inline(always)]
    fn with_locked<R>(&self, idx: usize, f: impl FnOnce(&mut HotSlot, ColdSlotPtrs) -> R) -> R {
        struct UnlockOnDrop<'a> {
            hp: &'a OnlineHashPrune,
            idx: usize,
        }
        impl Drop for UnlockOnDrop<'_> {
            fn drop(&mut self) {
                // SAFETY: idx bounds-checked before locking; we hold the lock.
                unsafe {
                    let hot_ptr = (self.hp.hot.as_ptr() as *mut HotSlot).add(self.idx);
                    (*hot_ptr).lock.unlock();
                }
            }
        }
        assert!(idx < self.hot.len(), "OnlineHashPrune index out of bounds");
        // SAFETY: bounds-checked; UnlockOnDrop unlocks on panic.
        let (hot_ptr, cold) = unsafe {
            let (hp, cold) = self.slot_ptrs(idx);
            (*hp).lock.lock();
            (hp, cold)
        };
        let _guard = UnlockOnDrop { hp: self, idx };
        unsafe { f(&mut *hot_ptr, cold) }
    }

    /// Online insert of edge `p -> c` (distance `dist`, precomputed `hash`).
    #[inline]
    pub fn add_edge(&self, p: usize, c: u32, hash: u16, dist: f32) {
        let l_max = self.l_max as u8;
        self.with_locked(p, |hot, cold| {
            // SAFETY: cold ptrs valid for scan_lanes; lock held.
            unsafe {
                insert_locked(hot, cold, hash, c, dist, l_max);
            }
        });
    }

    /// Drain node `p`'s reservoir, sorted by ascending distance, up to `max_degree`.
    pub fn get_neighbors(&self, p: usize, max_degree: usize) -> Vec<(u32, f32)> {
        self.with_locked(p, |hot, cold| {
            // SAFETY: ptrs valid for scan_lanes; lock held.
            unsafe { get_neighbors_saturated(hot, cold, max_degree) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_fill_then_evict_farthest() {
        let hp = OnlineHashPrune::new(4, 4);
        // node 2: four distinct buckets, distances 4,3,2,1
        hp.add_edge(2, 100, 10, 4.0);
        hp.add_edge(2, 200, 20, 3.0);
        hp.add_edge(2, 300, 30, 2.0);
        hp.add_edge(2, 400, 40, 1.0);
        // full; a closer candidate in a NEW bucket evicts the farthest (id 100 @4.0)
        hp.add_edge(2, 500, 50, 0.5);
        let got = hp.get_neighbors(2, 4);
        let ids: Vec<u32> = got.iter().map(|&(id, _)| id).collect();
        assert_eq!(got.len(), 4);
        assert!(!ids.contains(&100), "farthest evicted");
        assert!(ids.contains(&500));
        // other nodes are independent/empty
        assert!(hp.get_neighbors(0, 4).is_empty());
    }

    #[test]
    fn online_same_bucket_keeps_closer() {
        let hp = OnlineHashPrune::new(2, 8);
        hp.add_edge(1, 7, 99, 5.0);
        hp.add_edge(1, 8, 99, 2.0); // same bucket, closer -> replaces
        let got = hp.get_neighbors(1, 8);
        assert_eq!(got.len(), 1, "same bucket dedups");
        assert_eq!(got[0].0, 8);
    }

    #[test]
    fn online_concurrent_disjoint_nodes() {
        use std::sync::Arc;
        let hp = Arc::new(OnlineHashPrune::new(1000, 64));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let hp = hp.clone();
                std::thread::spawn(move || {
                    for n in (t..1000).step_by(8) {
                        for k in 0..50u32 {
                            hp.add_edge(n, n as u32 * 1000 + k, (k % 64) as u16, k as f32);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // each node saw 50 inserts across 64 buckets -> 50 distinct kept
        assert_eq!(hp.get_neighbors(500, 64).len(), 50);
    }
}
