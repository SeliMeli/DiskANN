/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::config::{self, MaxDegree};
use diskann_pipnn::{build_graph, PiPNNBuildContext, PiPNNConfig};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;

fn pipnn_config() -> PiPNNConfig {
    PiPNNConfig {
        c_max: 4,
        c_min: 1,
        p_samp: 0.5,
        fanout: vec![2],
        k: 1,
        replicas: 1,
    }
}

fn graph_config(metric: Metric, degree: usize) -> diskann::graph::Config {
    config::Builder::new_with(degree, MaxDegree::same(), 8, metric.into(), |builder| {
        builder.alpha(1.2);
    })
    .build()
    .unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn rows(graph: Vec<diskann::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
    graph.into_iter().map(Vec::from).collect()
}

#[test]
fn builds_a_single_leaf_graph_for_real_dataset_ids() {
    let data = [0.0_f32, 1.0, 2.0, 3.0];
    let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
    let graph = graph_config(Metric::L2, 2);
    let pool = pool(2);
    let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

    let actual = build_graph(data, &context).unwrap();

    assert_eq!(rows(actual), [vec![1], vec![0, 2], vec![1, 3], vec![2]]);
}

#[test]
fn rejects_empty_dataset_dimensions_at_the_public_boundary() {
    let graph = graph_config(Metric::L2, 2);
    let pool = pool(1);
    let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

    let no_rows = MatrixView::try_from(&[] as &[f32], 0, 4).unwrap();
    let no_columns = MatrixView::try_from(&[] as &[f32], 4, 0).unwrap();

    assert!(build_graph(no_rows, &context).is_err());
    assert!(build_graph(no_columns, &context).is_err());
}
