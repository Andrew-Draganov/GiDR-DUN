#ifndef UMAP_EXAMPLE_TEST_OBJECT_H
#define UMAP_EXAMPLE_TEST_OBJECT_H

void of();

void gpu_umap_wrap_old(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P, int epochs,
                   float lr, int neg_samples);

void
gpu_umap_head_tail_wrap(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_head, int *h_tail,
                        float *h_P, int epochs, float lrs);


void gpu_umap_wrap(
        int normalized,
        int sym_attraction,
        int momentum,
        float* head_embedding,
        float* tail_embedding,
        int* head,
        int* tail,
        float* weights,
        long* neighbor_counts,
        float* all_updates,
        float* gains,
        float a,
        float b,
        int dim,
        int n_vertices,
        float initial_lr,
        int n_edges,
        int n_epochs
);

#endif //UMAP_EXAMPLE_TEST_OBJECT_H
