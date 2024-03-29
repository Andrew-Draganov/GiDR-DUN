#ifndef GIDR_DUN_GPU_GRAPH_WEIGHTS_H
#define GIDR_DUN_GPU_GRAPH_WEIGHTS_H

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
);

#endif //GIDR_DUN_GPU_GRAPH_WEIGHTS_H
