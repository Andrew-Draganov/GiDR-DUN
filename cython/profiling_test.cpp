//
// Created by jakobrj on 3/11/22.
//

#import "gpu_kernels.h"

int main() {
    int normalized = 0;
    int sym_attraction = 0;
    int momentum = 0;
    float a = 1.576944;
    float b = 0.895061;
    int dim = 2;
    int n_vertices = 60000;
    float initial_lr = 0.1;
    int n_edges = n_vertices * 20;
    int n_epochs = 500;


    float *head_embedding = new float[n_vertices * dim]();
    float *tail_embedding = new float[n_vertices * dim]();
    int *head = new int[n_edges]();
    int *tail = new int[n_edges]();
    float *weights = new float[n_edges]();
    long *neighbor_counts = new long[n_vertices]();
    float *all_updates = new float[n_vertices * dim]();
    float *gains = new float[n_vertices * dim]();

    for (int i = 0; i < n_vertices; i++) {
        neighbor_counts[i] = 20;
    }

    gpu_umap(normalized,
             sym_attraction,
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
             n_epochs);

    return 0;
}