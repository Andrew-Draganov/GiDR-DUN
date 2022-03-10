#ifndef __gpu_kernels__
#define __gpu_kernels__

#include "cuda.h"
#include "cuda_runtime.h"

extern "C++" void gpuf();

extern "C++" void gpu_umap_old(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P, int epochs,
                           float lr, int neg_samples);

extern "C++" void
gpu_umap_head_tail(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_heads, int *h_tails,
                   float *h_P, int epochs, float init_lr);

extern "C++" void gpu_umap(
        int normalized,
        int sym_attraction,
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
        int n_epochs
);
#endif