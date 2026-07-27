/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use super::SortedNeighbors;
use crate::graph::config::PruneKind;

use crate::{
    ANNError, ANNErrorKind, error, graph::AdjacencyList, neighbor::Neighbor, utils::VectorId,
};

/// Options provided to prune. See the field-level documentation for more details.
///
/// This struct should be kept cheap to construct.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Options {
    /// Force adjacency list saturation.
    ///
    /// Adjacency list saturation expands the post-pruning candidate list up to the
    /// maximum degree by greedily adding skipped neighbors from the original candidate
    /// pool.
    pub(in crate::graph) force_saturate: bool,
}

/// An aggregate of scratch space used by the pruning algorithm for allocation.
///
/// The actual object passed to the pruning algorithms is [`Context`], which allows
/// sub-fields to be over-written as needed with local state if that is available instead.
#[derive(Debug)]
pub struct Scratch<I>
where
    I: VectorId,
{
    pub(in crate::graph) pool: Vec<Neighbor<I>>,
    pub(in crate::graph) available: Vec<bool>,
    pub(in crate::graph) states: Vec<State>,
    pub(in crate::graph) neighbors: AdjacencyList<I>,
}

impl<I> Scratch<I>
where
    I: VectorId,
{
    /// Create a new empty scratch space.
    ///
    /// This function should not allocate.
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            available: Vec::new(),
            states: Vec::new(),
            neighbors: AdjacencyList::new(),
        }
    }

    /// Convert `self` into a `Context`, truncating the internal `pool` list to a length of
    /// `max_candidates`.
    pub(in crate::graph) fn as_context(&mut self, max_candidates: usize) -> Context<'_, I> {
        Context {
            pool: SortedNeighbors::new(&mut self.pool, max_candidates),
            available: &mut self.available,
            states: &mut self.states,
            neighbors: &mut self.neighbors,
        }
    }

    /// Candidate buffer used by adapters before entering the synchronous prune kernel.
    pub fn candidates_mut(&mut self) -> &mut Vec<Neighbor<I>> {
        &mut self.pool
    }

    /// Most recent output produced by the synchronous prune kernel.
    pub fn neighbors(&self) -> &AdjacencyList<I> {
        &self.neighbors
    }
}

impl<I> Default for Scratch<I>
where
    I: VectorId,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments passed to the lowest-level pruning algorithm.
#[derive(Debug)]
pub(crate) struct Context<'ctx, I>
where
    I: VectorId,
{
    /// Input: The list of candidates to prune.
    pub(in crate::graph) pool: SortedNeighbors<'ctx, I>,
    /// Scratch: Availability of each positional candidate in the current provider view.
    pub(in crate::graph) available: &'ctx mut Vec<bool>,
    /// Scratch: State tracking for prune.
    pub(in crate::graph) states: &'ctx mut Vec<State>,
    /// Output: The pruned candidates list.
    pub(in crate::graph) neighbors: &'ctx mut AdjacencyList<I>,
}

/// Position-wise state tracking.
///
/// Refer to the inline documentation in [`DiskANNIndex::occlude_list`] for documentation
/// on the use of these fields.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State {
    /// The occlude factor for the pool item at the corresponding index.
    pub(in crate::graph) occlude_factor: f32,
    /// The index of the last checked neighbor.
    pub(in crate::graph) last_checked: u16,
    /// The candidate index of this neighbor.
    pub(in crate::graph) neighbor: u16,
}

/// Provider-independent settings for Vamana's RobustPrune state machine.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    degree: usize,
    alpha: f32,
    prune_kind: PruneKind,
    saturate: bool,
}

impl Policy {
    pub fn new(degree: usize, alpha: f32, prune_kind: PruneKind, saturate: bool) -> Self {
        Self {
            degree,
            alpha,
            prune_kind,
            saturate,
        }
    }
}

/// Errors returned by [`robust_prune`].
#[derive(Debug, Error)]
pub enum RobustPruneError<E = std::convert::Infallible> {
    #[error("robust prune alpha must be finite and at least 1.0, got {0}")]
    InvalidAlpha(f32),
    #[error("robust prune supports at most {max} candidates, got {actual}")]
    TooManyCandidates { actual: usize, max: usize },
    #[error("failed to reserve robust-prune workspace: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("distance computation failed: {0}")]
    Distance(E),
}

/// Run Vamana's synchronous RobustPrune state machine.
///
/// The caller fills [`Scratch::candidates_mut`] in source-distance order. Provider fills
/// and missing-vector policy stay in the outer adapter; this kernel caches only stable
/// positional availability so provider borrows never escape a call.
pub fn robust_prune<I, E, D, A, X>(
    scratch: &mut Scratch<I>,
    policy: Policy,
    distance: D,
    is_available: A,
    exclude: X,
) -> Result<(), RobustPruneError<E>>
where
    I: VectorId,
    D: FnMut(I, I) -> Result<f32, E>,
    A: FnMut(I) -> bool,
    X: Fn(I) -> bool,
{
    let Scratch {
        pool,
        available,
        states,
        neighbors,
    } = scratch;
    robust_prune_parts(
        Workspace {
            candidates: pool,
            available,
            states,
            neighbors,
        },
        policy,
        distance,
        is_available,
        exclude,
    )
}

pub(in crate::graph) struct Workspace<'a, I: VectorId> {
    pub(in crate::graph) candidates: &'a [Neighbor<I>],
    pub(in crate::graph) available: &'a mut Vec<bool>,
    pub(in crate::graph) states: &'a mut Vec<State>,
    pub(in crate::graph) neighbors: &'a mut AdjacencyList<I>,
}

pub(in crate::graph) fn robust_prune_parts<I, E, D, A, X>(
    workspace: Workspace<'_, I>,
    policy: Policy,
    mut distance: D,
    mut is_available: A,
    exclude: X,
) -> Result<(), RobustPruneError<E>>
where
    I: VectorId,
    D: FnMut(I, I) -> Result<f32, E>,
    A: FnMut(I) -> bool,
    X: Fn(I) -> bool,
{
    let Workspace {
        candidates,
        available,
        states,
        neighbors,
    } = workspace;

    if !policy.alpha.is_finite() || policy.alpha < 1.0 {
        return Err(RobustPruneError::InvalidAlpha(policy.alpha));
    }
    if candidates.len() > u16::MAX as usize {
        return Err(RobustPruneError::TooManyCandidates {
            actual: candidates.len(),
            max: u16::MAX as usize,
        });
    }
    if candidates.is_empty() {
        neighbors.clear();
        return Ok(());
    }

    available
        .try_reserve(candidates.len().saturating_sub(available.len()))
        .map_err(RobustPruneError::Allocation)?;
    available.clear();
    available.extend(
        candidates
            .iter()
            .map(|candidate| !exclude(candidate.id) && is_available(candidate.id)),
    );

    states
        .try_reserve(candidates.len().saturating_sub(states.len()))
        .map_err(RobustPruneError::Allocation)?;
    let output_capacity = policy.degree.min(candidates.len());
    neighbors
        .try_reserve(output_capacity.saturating_sub(neighbors.len()))
        .map_err(RobustPruneError::Allocation)?;
    states.clear();
    states.resize(candidates.len(), State::default());
    std::iter::zip(states.iter_mut(), available.iter()).for_each(|(state, available)| {
        if !available {
            state.occlude_factor = f32::MAX;
        }
    });

    let mut current_alpha = 1.0f32;
    let increment_factor = policy.alpha.min(1.2);
    let mut found = 0;

    while found < policy.degree {
        for (i, candidate) in candidates.iter().enumerate() {
            if found >= policy.degree {
                break;
            }

            let State {
                mut occlude_factor,
                mut last_checked,
                ..
            } = states[i];

            if !available[i] {
                states[i].occlude_factor = f32::MAX;
                continue;
            }
            if occlude_factor == f32::MAX || occlude_factor > current_alpha {
                continue;
            }

            while last_checked as usize != found {
                let result_position = states[last_checked as usize].neighbor as usize;
                last_checked += 1;

                if result_position >= i {
                    states[i].last_checked = last_checked;
                    continue;
                }

                let pair_distance = distance(candidate.id, candidates[result_position].id)
                    .map_err(RobustPruneError::Distance)?;

                occlude_factor = policy.prune_kind.update_occlude_factor(
                    candidate.distance,
                    pair_distance,
                    occlude_factor,
                    current_alpha,
                );

                if occlude_factor > current_alpha {
                    break;
                }
            }

            let state = &mut states[i];
            state.last_checked = last_checked;
            if occlude_factor > current_alpha {
                state.occlude_factor = occlude_factor;
                continue;
            }

            state.occlude_factor = f32::MAX;
            states[found].neighbor = i as u16;
            found += 1;
        }

        if current_alpha == policy.alpha {
            break;
        }
        current_alpha = (current_alpha * increment_factor).min(policy.alpha);
    }

    let mut guard = neighbors
        .try_resize(found)
        .map_err(RobustPruneError::Allocation)?;
    std::iter::zip(guard.iter_mut(), states.iter())
        .for_each(|(destination, state)| *destination = candidates[state.neighbor as usize].id);
    guard.finish(found);

    if policy.saturate {
        for candidate in candidates {
            if neighbors.len() >= policy.degree {
                break;
            }
            if !exclude(candidate.id) && is_available(candidate.id) {
                neighbors.push(candidate.id);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct DistanceFailure;

    fn scratch_with_candidates(count: usize) -> Scratch<u32> {
        let mut scratch = Scratch::new();
        scratch.candidates_mut().extend(
            (0..count).map(|id| Neighbor::new(u32::try_from(id).unwrap_or(u32::MAX), id as f32)),
        );
        scratch
    }

    #[test]
    fn rejects_invalid_alpha() {
        let mut scratch = Scratch::<u32>::new();
        let error = robust_prune(
            &mut scratch,
            Policy::new(1, f32::NAN, PruneKind::TriangleInequality, false),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| true,
            |_| false,
        )
        .unwrap_err();

        assert!(matches!(error, RobustPruneError::InvalidAlpha(value) if value.is_nan()));
    }

    #[test]
    fn rejects_too_many_candidates() {
        let mut scratch = scratch_with_candidates(u16::MAX as usize + 1);
        let error = robust_prune(
            &mut scratch,
            Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| true,
            |_| false,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RobustPruneError::TooManyCandidates {
                actual: 65_536,
                max: 65_535
            }
        ));
    }

    #[test]
    fn propagates_distance_failure() {
        let mut scratch = scratch_with_candidates(2);
        let error = robust_prune(
            &mut scratch,
            Policy::new(2, 1.2, PruneKind::TriangleInequality, false),
            |_, _| Err(DistanceFailure),
            |_| true,
            |_| false,
        )
        .unwrap_err();

        assert!(matches!(error, RobustPruneError::Distance(DistanceFailure)));
    }

    #[test]
    fn empty_candidates_clear_previous_output() {
        let mut scratch = Scratch::<u32>::new();
        scratch.neighbors.push(7);

        robust_prune(
            &mut scratch,
            Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| true,
            |_| false,
        )
        .unwrap();

        assert!(scratch.neighbors().is_empty());
    }

    #[test]
    fn excludes_candidate_even_when_its_vector_is_cached() {
        let mut scratch = scratch_with_candidates(2);

        robust_prune(
            &mut scratch,
            Policy::new(1, 1.2, PruneKind::TriangleInequality, false),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| true,
            |id| id == 0,
        )
        .unwrap();

        assert_eq!(scratch.neighbors().as_ref(), &[1]);
    }

    #[test]
    fn saturation_excludes_unavailable_candidates() {
        let mut scratch = scratch_with_candidates(3);

        robust_prune(
            &mut scratch,
            Policy::new(3, 1.2, PruneKind::TriangleInequality, true),
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |id| id != 1,
            |_| false,
        )
        .unwrap();

        assert_eq!(scratch.neighbors().as_ref(), &[0, 2]);
    }

    #[test]
    fn reuses_positional_workspace() {
        let mut scratch = scratch_with_candidates(3);
        let policy = Policy::new(2, 1.2, PruneKind::TriangleInequality, false);

        robust_prune(
            &mut scratch,
            policy,
            |_, _| Ok::<_, std::convert::Infallible>(1.0),
            |_| true,
            |_| false,
        )
        .unwrap();
        let capacities = (
            scratch.pool.capacity(),
            scratch.available.capacity(),
            scratch.states.capacity(),
            scratch.neighbors.capacity(),
        );

        robust_prune(
            &mut scratch,
            policy,
            |_, _| Ok::<_, std::convert::Infallible>(1.0),
            |_| true,
            |_| false,
        )
        .unwrap();

        assert_eq!(
            (
                scratch.pool.capacity(),
                scratch.available.capacity(),
                scratch.states.capacity(),
                scratch.neighbors.capacity(),
            ),
            capacities
        );
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("retrieval of main vector id {} failed during prune aggregation", self.0)]
pub(crate) struct FailedVectorRetrieval<I>(I)
where
    I: VectorId;

impl<I> error::TransientError<ANNError> for FailedVectorRetrieval<I>
where
    I: VectorId,
{
    fn acknowledge<D>(self, _why: D)
    where
        D: std::fmt::Display,
    {
    }

    #[track_caller]
    #[inline(never)]
    fn escalate<D>(self, why: D) -> ANNError
    where
        D: std::fmt::Display,
    {
        ANNError::new(ANNErrorKind::IndexError, self).context(why.to_string())
    }
}

/// Failure condition for [`DiskANNIndex::robust_prune_list`].
///
/// It's currently possible for retrieval of the id being pruned to fail due to a transient
/// error. We do not always want to escalate this as a hard error, and thus provide an
/// option for transient error handling.
#[derive(Debug)]
pub(crate) enum ListError<I>
where
    I: VectorId,
{
    /// A potentially transient error.
    FailedVectorRetrieval(FailedVectorRetrieval<I>),
    /// A critical error.
    Other(ANNError),
}

impl<I> ListError<I>
where
    I: VectorId,
{
    pub(in crate::graph) fn failed_retrieval(id: I) -> Self {
        Self::FailedVectorRetrieval(FailedVectorRetrieval(id))
    }
}

impl<I> From<ANNError> for ListError<I>
where
    I: VectorId,
{
    fn from(err: ANNError) -> Self {
        Self::Other(err)
    }
}

impl<I> error::ToRanked for ListError<I>
where
    I: VectorId,
{
    type Transient = FailedVectorRetrieval<I>;
    type Error = ANNError;

    fn to_ranked(self) -> error::RankedError<Self::Transient, Self::Error> {
        match self {
            Self::FailedVectorRetrieval(err) => error::RankedError::Transient(err),
            Self::Other(err) => error::RankedError::Error(err),
        }
    }

    fn from_transient(transient: Self::Transient) -> Self {
        Self::FailedVectorRetrieval(transient)
    }

    fn from_error(error: Self::Error) -> Self {
        Self::Other(error)
    }
}
