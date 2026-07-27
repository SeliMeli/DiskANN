/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_pipnn::partition::{
    nearest_leaders, PartitionKernelError, PartitionTopK, MAX_PARTITION_FANOUT,
};
use diskann_vector::distance::Metric;

#[test]
fn l2_keeps_the_first_leader_when_boundary_distances_tie() {
    #[rustfmt::skip]
    let dots = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 4.0, 6.0,
    ];
    let leader_squared_norms = [0.0, 1.0, 4.0, 9.0];
    let mut assignments = [u32::MAX; 4];

    let input = PartitionTopK {
        dots: &dots,
        rows: 2,
        leaders: 4,
        row_scales: &[],
        leader_scales: &leader_squared_norms,
        metric: Metric::L2,
    };

    nearest_leaders(input, 2, &mut assignments).unwrap();

    assert_eq!(assignments, [0, 1, 2, 1]);
}

#[test]
fn supports_every_partition_metric() {
    #[rustfmt::skip]
    let dots = [
        1.0, 0.0, -1.0,
        2.0, 6.0, 0.0,
    ];

    let cases = [
        (Metric::L2, &[][..], &[1.0, 4.0, 9.0][..], [0, 1, 1, 0]),
        (
            Metric::Cosine,
            &[1.0, 4.0][..],
            &[1.0, 2.0, 3.0][..],
            [0, 1, 1, 0],
        ),
        (Metric::CosineNormalized, &[][..], &[][..], [0, 1, 1, 0]),
        (Metric::InnerProduct, &[][..], &[][..], [0, 1, 1, 0]),
    ];

    for (metric, row_scales, leader_scales, expected) in cases {
        let mut assignments = [u32::MAX; 4];
        nearest_leaders(
            PartitionTopK {
                dots: &dots,
                rows: 2,
                leaders: 3,
                row_scales,
                leader_scales,
                metric,
            },
            2,
            &mut assignments,
        )
        .unwrap();

        assert_eq!(assignments, expected, "metric {metric:?}");
    }
}

#[test]
fn cosine_treats_a_zero_norm_as_zero_similarity() {
    let mut assignments = [u32::MAX; 2];

    nearest_leaders(
        PartitionTopK {
            dots: &[100.0, -100.0],
            rows: 1,
            leaders: 2,
            row_scales: &[0.0],
            leader_scales: &[1.0, 1.0],
            metric: Metric::Cosine,
        },
        2,
        &mut assignments,
    )
    .unwrap();

    assert_eq!(assignments, [0, 1]);
}

#[test]
fn ignores_nan_distances_without_displacing_finite_leaders() {
    let mut assignments = [u32::MAX; 2];

    nearest_leaders(
        PartitionTopK {
            dots: &[f32::NAN, 3.0, 2.0],
            rows: 1,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut assignments,
    )
    .unwrap();

    assert_eq!(assignments, [1, 2]);
}

#[test]
fn rejects_rows_with_too_few_rankable_distances() {
    let error = nearest_leaders(
        PartitionTopK {
            dots: &[f32::NAN, 3.0],
            rows: 1,
            leaders: 2,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut [u32::MAX; 2],
    )
    .unwrap_err();

    assert_eq!(
        error,
        PartitionKernelError::InsufficientRankableDistances { row: 0, fanout: 2 }
    );
}

#[test]
fn accepts_empty_rows_and_zero_fanout() {
    nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: 0,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut [],
    )
    .unwrap();

    nearest_leaders(
        PartitionTopK {
            dots: &[1.0, 2.0, 3.0],
            rows: 1,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
    )
    .unwrap();

    // `u32::MAX` leaders still have positions representable by `u32`: the
    // largest position is `u32::MAX - 1`. An empty batch lets us exercise the
    // validation boundary without allocating the declared tile.
    nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: 0,
            leaders: u32::MAX as usize,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
    )
    .unwrap();
}

#[test]
fn rejects_inconsistent_shapes_and_fanout() {
    let base = PartitionTopK {
        dots: &[0.0; 6],
        rows: 2,
        leaders: 3,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };

    assert_eq!(
        nearest_leaders(
            PartitionTopK {
                dots: &[0.0; 5],
                ..base
            },
            2,
            &mut [0; 4],
        ),
        Err(PartitionKernelError::InvalidBufferLength {
            buffer: "dot-product tile",
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(
        nearest_leaders(base, 2, &mut [0; 3]),
        Err(PartitionKernelError::InvalidBufferLength {
            buffer: "output",
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(
        nearest_leaders(base, MAX_PARTITION_FANOUT + 1, &mut []),
        Err(PartitionKernelError::InvalidFanout {
            fanout: MAX_PARTITION_FANOUT + 1,
            leaders: 3,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let one_leader = PartitionTopK {
        dots: &[0.0],
        rows: 1,
        leaders: 1,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };
    assert_eq!(
        nearest_leaders(one_leader, 2, &mut []),
        Err(PartitionKernelError::InvalidFanout {
            fanout: 2,
            leaders: 1,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let exact_maximum = PartitionTopK {
        dots: &[],
        rows: 0,
        leaders: MAX_PARTITION_FANOUT,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };
    nearest_leaders(exact_maximum, MAX_PARTITION_FANOUT, &mut []).unwrap();
}

#[test]
fn rejects_shape_overflow_before_reading_buffers() {
    let error = nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: usize::MAX,
            leaders: 2,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        1,
        &mut [],
    )
    .unwrap_err();

    assert_eq!(
        error,
        PartitionKernelError::ShapeOverflow {
            buffer: "dot-product tile",
            rows: usize::MAX,
            cols: 2,
        }
    );
}
