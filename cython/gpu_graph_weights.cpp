//
// Created by jakobrj on 5/4/22.
//
#import <stdio.h>
#include "gpu_graph_cuda.cuh"
#include "gpu_graph_weights.h"

void gpu_graph_weights() {
    printf("hello from the object file!\n");
    cuda();
}


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
            local_connectivity,
            return_dists,
            pseudo_distance
    );
}