/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_pipnn::leaf::{
    nearest_leaf_neighbors, LeafKernelError, LeafNeighbor, LeafTopK, LeafTopKWorkspace,
};
use diskann_vector::distance::Metric;

fn run(dots: &[f32], points: usize, k: usize, metric: Metric) -> (usize, Vec<LeafNeighbor>) {
    let actual_k = k.min(points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); points * actual_k];
    let mut workspace = LeafTopKWorkspace::new();
    let returned_k = nearest_leaf_neighbors(
        LeafTopK {
            dots,
            points,
            metric,
        },
        k,
        &mut output,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(returned_k, actual_k);
    (returned_k, output)
}

#[test]
fn l2_scans_only_the_lower_triangle_and_breaks_ties_by_position() {
    #[rustfmt::skip]
    let dots = [
        0.0, 999.0, 999.0, 999.0,
        0.0,   1.0, 999.0, 999.0,
        0.0,   0.0,   1.0, 999.0,
        0.0,   1.0,   1.0,   2.0,
    ];

    let (_, output) = run(&dots, 4, 2, Metric::L2);

    assert_eq!(
        output,
        [
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(2, 1.0),
            LeafNeighbor::new(0, 1.0),
            LeafNeighbor::new(3, 1.0),
            LeafNeighbor::new(0, 1.0),
            LeafNeighbor::new(3, 1.0),
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(2, 1.0),
        ]
    );
}

#[test]
fn supports_every_leaf_metric() {
    #[rustfmt::skip]
    let dots = [
        1.0, 77.0, 77.0,
        0.0,  1.0, 77.0,
       -1.0,  0.5,  1.0,
    ];

    let cases = [
        (Metric::L2, [1, 2, 1]),
        (Metric::Cosine, [1, 2, 1]),
        (Metric::CosineNormalized, [1, 2, 1]),
        (Metric::InnerProduct, [1, 2, 1]),
    ];

    for (metric, expected) in cases {
        let (_, output) = run(&dots, 3, 1, metric);
        let positions: Vec<_> = output.iter().map(|neighbor| neighbor.position).collect();
        assert_eq!(positions, expected, "metric {metric:?}");
    }
}

#[test]
fn cosine_treats_zero_norm_as_zero_similarity() {
    #[rustfmt::skip]
    let dots = [
        0.0, 11.0, 11.0,
        0.0,  1.0, 11.0,
        0.0,  0.0,  1.0,
    ];

    let (_, output) = run(&dots, 3, 2, Metric::Cosine);

    assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
    assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
}

#[test]
fn preserves_pipnn_metric_edge_semantics() {
    #[rustfmt::skip]
    let out_of_range = [
        1.0, 0.0,
        2.0, 1.0,
    ];
    assert_eq!(run(&out_of_range, 2, 1, Metric::L2).1[0].distance, 0.0);
    assert_eq!(
        run(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
        0.0
    );
    assert_eq!(run(&out_of_range, 2, 1, Metric::Cosine).1[0].distance, 0.0);

    #[rustfmt::skip]
    let opposite = [
         1.0, 0.0,
        -2.0, 1.0,
    ];
    assert_eq!(run(&opposite, 2, 1, Metric::Cosine).1[0].distance, 3.0);

    let subnormal_squared_norm = f32::MIN_POSITIVE / 2.0;
    #[rustfmt::skip]
    let subnormal = [
        subnormal_squared_norm, 0.0,
        1.0,                    1.0,
    ];
    assert_eq!(run(&subnormal, 2, 1, Metric::Cosine).1[0].distance, 1.0);

    let minimum_normal_squared_norm = f32::MIN_POSITIVE;
    #[rustfmt::skip]
    let minimum_normal = [
        minimum_normal_squared_norm,          0.0,
        minimum_normal_squared_norm.sqrt(),   1.0,
    ];
    assert_eq!(
        run(&minimum_normal, 2, 1, Metric::Cosine).1[0].distance,
        0.0
    );
}

#[test]
fn every_metric_ignores_nan_pairs() {
    #[rustfmt::skip]
    let dots = [
        1.0,       0.0, 0.0,
        f32::NAN,  1.0, 0.0,
        0.5,       0.25, 1.0,
    ];

    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let (_, output) = run(&dots, 3, 1, metric);
        assert_eq!(output[0].position, 2, "metric {metric:?}");
        assert_eq!(output[1].position, 2, "metric {metric:?}");
    }
}

#[test]
fn rejects_incomplete_neighbor_rows() {
    #[rustfmt::skip]
    let dots = [
        1.0,      0.0,
        f32::NAN, 1.0,
    ];
    let mut output = [LeafNeighbor::default(); 2];
    let mut workspace = LeafTopKWorkspace::new();

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &dots,
            points: 2,
            metric: Metric::L2,
        },
        1,
        &mut output,
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InsufficientRankableNeighbors {
            row: 0,
            neighbors: 1,
        }
    );
}

#[test]
fn clamps_k_to_available_non_self_neighbors() {
    #[rustfmt::skip]
    let dots = [
        1.0, 3.0, 3.0,
        0.0, 1.0, 3.0,
        0.0, 0.0, 1.0,
    ];

    let (actual_k, output) = run(&dots, 3, 99, Metric::L2);

    assert_eq!(actual_k, 2);
    assert_eq!(output.len(), 6);
    for (row, neighbors) in output.chunks_exact(actual_k).enumerate() {
        assert!(neighbors
            .iter()
            .all(|neighbor| neighbor.position as usize != row));
    }
}

#[test]
fn accepts_empty_singleton_and_zero_k_inputs() {
    let mut workspace = LeafTopKWorkspace::new();
    let empty = LeafTopK {
        dots: &[],
        points: 0,
        metric: Metric::L2,
    };
    assert_eq!(
        nearest_leaf_neighbors(empty, 2, &mut [], &mut workspace).unwrap(),
        0
    );

    let singleton = LeafTopK {
        dots: &[4.0],
        points: 1,
        metric: Metric::Cosine,
    };
    assert_eq!(
        nearest_leaf_neighbors(singleton, 2, &mut [], &mut workspace).unwrap(),
        0
    );

    let pair = LeafTopK {
        dots: &[1.0, 0.0, 0.0, 1.0],
        points: 2,
        metric: Metric::InnerProduct,
    };
    assert_eq!(
        nearest_leaf_neighbors(pair, 0, &mut [], &mut workspace).unwrap(),
        0
    );
}

#[test]
fn rejects_invalid_shapes_before_dispatch() {
    let mut workspace = LeafTopKWorkspace::new();
    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[0.0; 8],
            points: 3,
            metric: Metric::L2,
        },
        1,
        &mut [LeafNeighbor::default(); 3],
        &mut workspace,
    )
    .unwrap_err();
    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "lower dot-product matrix",
            expected: 9,
            actual: 8,
        }
    );

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[0.0; 9],
            points: 3,
            metric: Metric::L2,
        },
        2,
        &mut [LeafNeighbor::default(); 5],
        &mut workspace,
    )
    .unwrap_err();
    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "output",
            expected: 6,
            actual: 5,
        }
    );
}

#[test]
fn rejects_shape_overflow_before_reading_buffers() {
    let mut workspace = LeafTopKWorkspace::new();
    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[],
            points: usize::MAX,
            metric: Metric::L2,
        },
        1,
        &mut [],
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(error, LeafKernelError::TooManyPoints(usize::MAX));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn accepts_the_largest_representable_point_count_before_shape_validation() {
    let points = u32::MAX as usize;
    let expected = points.checked_mul(points).unwrap();
    let mut workspace = LeafTopKWorkspace::new();

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[],
            points,
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "lower dot-product matrix",
            expected,
            actual: 0,
        }
    );
}
