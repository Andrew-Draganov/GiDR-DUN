#import "../cuda_kernels/gpu_kernels.h"
#import "../cuda_wrappers/gpu_dim_reduction.h"

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
        int n_epochs,
        int negative_sample_rate
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
            n_epochs,
            negative_sample_rate
    );
}

void GPU_KNN_wrap(int *h_neighbors, float *h_distances, float *h_data, int n, int d, int k){
    GPU_KNN(h_neighbors,h_distances,h_data, n, d, k);
}
