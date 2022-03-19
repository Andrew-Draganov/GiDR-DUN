#import "gpu_kernels.h"
#import "gpu_dim_reduction.h"

#import <stdio.h>

void of() {
    printf("hello from the object file!\n");
    gpuf();
}

//void gpu_umap_wrap_old(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P,
//                   int epochs, float lr, int neg_samples) {
//    gpu_umap_old(n, h_D_embed, dims_embed, h_N, k, h_P, epochs, lr, neg_samples);
//}


//void gpu_umap_head_tail_wrap(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_head, int *h_tail,
//                             float *h_P, int epochs, float lr) {
//    gpu_umap_head_tail(n_edges, n_vertices, h_D_embed, dims_embed, h_head, h_tail, h_P, epochs, lr);
//}


void gpu_umap_wrap(
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
        int n_epochs
) {
    gpu_umap(
            normalized,
            sym_attraction,
            frob,
            momentum,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            neighbor_counts,
            all_updates,
            gains,
            a,
            b,
            dim,
            n_vertices,
            initial_lr,
            n_edges,
            n_epochs
    );
}