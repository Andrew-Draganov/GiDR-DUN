#import "../cuda_kernels/gpu_kernels.h"
#import "../cuda_wrappers/gpu_dim_reduction.h"

void gpu_wrapper(
        int normalized,
        int sym_attraction,
        int frob,
        int momentum,
        float *head_embedding,
        int *head,
        float *weights,
        long *neighbor_counts,
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
    gpu_gdr(
            normalized,
            sym_attraction,
            frob,
            momentum,
            head_embedding,
            head,
            weights,
            neighbor_counts,
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
