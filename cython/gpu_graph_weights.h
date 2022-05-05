//
// Created by jakobrj on 5/4/22.
//

#ifndef GIDR_DUN_GPU_GRAPH_WEIGHTS_H
#define GIDR_DUN_GPU_GRAPH_WEIGHTS_H

void gpu_graph_weights();

void compute_neighbor_graph(
        int *rows, //return val
        int *cols, //return val
        float *values, //return val
        float *dists, //return val
        float *sigmas, //return val
        float *rhos, //return val
        int *knn_indices,
        float *knn_dists,
        int n_points,
        int n_neighbors,
        float local_connectivity,
        int return_dists,
        int pseudo_distance
);

#endif //GIDR_DUN_GPU_GRAPH_WEIGHTS_H
