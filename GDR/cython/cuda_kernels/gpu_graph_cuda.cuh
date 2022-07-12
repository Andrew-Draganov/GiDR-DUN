//
// Created by jakobrj on 5/4/22.
//

#ifndef GIDR_DUN_GPU_GRAPH_CUDA_CUH
#define GIDR_DUN_GPU_GRAPH_CUDA_CUH


void compute_neighbor_graph_cuda(
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
);

#endif //GIDR_DUN_GPU_GRAPH_CUDA_CUH
