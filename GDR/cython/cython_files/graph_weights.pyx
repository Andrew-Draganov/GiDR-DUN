cimport numpy as np
cimport cython

np.import_array()

cdef extern from "../cuda_wrappers/gpu_graph_weights.cpp":
    void compute_neighbor_graph(
        int*rows, #return val
        int*cols, #return val
        float *values, #return val
        float *dists, #return val
        float *sigmas, #return val
        float *rhos, #return val
        int *knn_indices,
        float *knn_dists,
        int n_points,
        int n_neighbors,
        float local_connectivity,
        int return_dists,
        int pseudo_distance
    )

cdef void _graph_weights(
    float[:] sigmas,
    float[:] rhos,
    int[:] rows,
    int[:] cols,
    float[:] values,
    float[:] dists,
    int[:, :] knn_indices,
    float[:, :] knn_dists,
    int n_points,
    int n_neighbors,
    int return_dists,
    float local_connectivity,
    int pseudo_distance
):
    compute_neighbor_graph(
        &rows[0], #return val
        &cols[0], #return val
        &values[0], #return val
        &dists[0], #return val
        &sigmas[0], #return val
        &rhos[0], #return val
        &knn_indices[0,0],
        &knn_dists[0,0],
        n_points,
        n_neighbors,
        local_connectivity,
        return_dists,
        pseudo_distance
    )

def graph_weights(
        float[:] sigmas,
        float[:] rhos,
        int[:] rows,
        int[:] cols,
        float[:] values,
        float[:] dists,
        int[:, :] knn_indices,
        float[:, :] knn_dists,
        int n_neighbors,
        int return_dists,
        float local_connectivity,
        int pseudo_distance
):
    cdef int n_points = knn_dists.shape[0]
    _graph_weights(
        sigmas,
        rhos,
        rows,
        cols,
        values,
        dists,
        knn_indices,
        knn_dists,
        n_points,
        n_neighbors,
        return_dists,
        local_connectivity,
        pseudo_distance
    )
