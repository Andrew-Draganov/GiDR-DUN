#import "gpu_kernels.h"
#import "gpu_dim_reduction.h"

#import <stdio.h>

void of() {
    printf("hello from the object file!\n");
    gpuf();
}

void gpu_umap_wrap(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P,
                   int epochs, float lr, int neg_samples) {
    gpu_umap(n, h_D_embed, dims_embed, h_N, k, h_P, epochs, lr, neg_samples);
}


void gpu_umap_head_tail_wrap(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_head, int *h_tail,
                             float *h_P, int epochs, float lr) {
    gpu_umap_head_tail(n_edges, n_vertices, h_D_embed, dims_embed, h_head, h_tail, h_P, epochs, lr);
}