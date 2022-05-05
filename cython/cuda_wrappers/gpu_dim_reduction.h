#ifndef UMAP_EXAMPLE_TEST_OBJECT_H
#define UMAP_EXAMPLE_TEST_OBJECT_H

void gpu_umap_wrap(
        int normalized,
        int sym_attraction,
        int frob,
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
        int n_epochs,
        int negative_sample_rate
);

#endif //UMAP_EXAMPLE_TEST_OBJECT_H
