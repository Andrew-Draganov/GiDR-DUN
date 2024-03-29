#import <stdio.h>
#include "../cuda_kernels/gpu_graph_cuda.cuh"
#include "../cuda_wrappers/gpu_graph_weights.h"

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
        int pseudo_distance
) {

    compute_neighbor_graph_cuda(
            rows, //return val
            cols, //return val
            values, //return val
            dists, //return val
            sigmas, //return val
            rhos, //return val
            knn_indices,
            knn_dists,
            n_points,
            n_neighbors,
            pseudo_distance
    );
}
