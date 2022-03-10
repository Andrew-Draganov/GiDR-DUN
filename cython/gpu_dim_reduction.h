#ifndef UMAP_EXAMPLE_TEST_OBJECT_H
#define UMAP_EXAMPLE_TEST_OBJECT_H

void of();

void gpu_umap_wrap(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P, int epochs,
                   float lr, int neg_samples);

void
gpu_umap_head_tail_wrap(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_head, int *h_tail,
                        float *h_P, int epochs, float lrs);

#endif //UMAP_EXAMPLE_TEST_OBJECT_H
