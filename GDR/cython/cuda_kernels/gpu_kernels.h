#ifndef __gpu_kernels__
#define __gpu_kernels__

#include "cuda.h"
#include "cuda_runtime.h"

extern "C++" void gpuf();

extern "C++" void gpu_umap(
        int normalized,
        int sym_attraction,
        int frob,
        int momentum,
        float *head_embedding,
        float *tail_embedding,
        int *head,
        int *tail,
        float *weights,
        long *neighbor_counts,
        float *all_updates,
        float *gains,
        float a,
        float b,
        int dim,
        int n_vertices,
        float initial_lr,
        int n_edges,
        int n_epochs,
        int negative_sample_rate
);

extern "C++" void GPU_KNN(int *h_neighbors, float *h_distances, float *h_data, int n, int d, int k);
#endif
