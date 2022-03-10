#import <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#import "gpu_kernels.h"
#import "GPU_utils.cuh"


__global__
void kernel() {
    printf("hello from the kernel!\n");
}

void gpuf() {
    printf("hello from the gpu file!\n");
    cudaDeviceSynchronize();
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#define BLOCK_SIZE 1024

/// https://github.com/rapidsai/cuml/blob/branch-22.04/cpp/src/umap/runner.cuh

__device__
float sqrd_dist(float *d_D, int dims, int i, int j) {
    float distance = 0.;
    for (int l = 0; l < dims; l++) {
        float diff = d_D[i * dims + l] - d_D[j * dims + l];
        distance += diff * diff;
    }
    return distance;
}

__device__
float q(float distance) {
    return 1 / (1 + distance * distance);
}

__global__
void init_random(curandState *d_random) {
    //initialize d_random
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int seed = id; // different seed per thread
    curand_init(seed, id, 0, &d_random[id]);
}

/// only for k = 1
//__global__
//void compute_grads(float *d_grads, float *d_P, int n, int *d_N, int k, float *d_D_embed,
//                   int dims_embed, float lr, curandState *d_random) {
//    int id = threadIdx.x + blockDim.x * blockIdx.x;
//
//    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
//        for (int l = 0; l < k; l++) {
//            int j = d_N[i * k + l];
//
//            int g = curand(&d_random[id]) % n;//random int
//
//            float attr = q(dist(d_D_embed, dims_embed, i, j));
//            attr = attr * attr * d_P[i * k + j];
//
//            float rep = q(dist(d_D_embed, dims_embed, i, g));
//            rep = rep * rep * rep;
//
//            for (int h = 0; h < dims_embed; h++) {
//                d_grads[i * dims_embed + h] +=
//                        lr * (attr * (d_D_embed[i * dims_embed + h] - d_D_embed[j * dims_embed + h]) -
//                              rep * (d_D_embed[i * dims_embed + h] - d_D_embed[g * dims_embed + h]));
//            }
//        }
//    }
//}

__device__
int get_start(int *d_ends, int i) {
    return i == 0 ? 0 : d_ends[i - 1];
}

__device__
int get_end(int *d_ends, int i) {
    return d_ends[i];
}

__device__
double fast_pow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = {a};
    if (b == 1.0) {
        return a;
    }
    u.x[1] = (int) (b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

__device__
float umap_attraction_grad(float dist_squared, float a, float b) {
    float grad_scalar = 0.0;
    grad_scalar = 2.0 * a * b * fast_pow(dist_squared, b - 1.0);
    grad_scalar /= a * fast_pow(dist_squared, b) + 1.0;
    return grad_scalar;
}

__device__
float kernel_function(float dist_squared, float a, float b) {
    if (b <= 1)
        return 1 / (1 + a * fast_pow(dist_squared, b));
    return fast_pow(dist_squared, b - 1) / (1 + a * fast_pow(dist_squared, b));
}

__device__
float attractive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
) {
    float edge_force;
    if (normalized == 0)
        edge_force = umap_attraction_grad(dist_squared, a, b);
    else
        edge_force = kernel_function(dist_squared, a, b);

    return edge_force * edge_weight;
}

// for any k
__global__
void compute_grads(float *d_grads, float *d_weight, int n, int *d_N, int *d_neighbor_ends, float *d_D_embed,
                   int dims_embed, curandState *d_random) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int l = get_start(d_neighbor_ends, i); l < get_end(d_neighbor_ends, i); l++) {
            int j = d_N[l];
            float attr = q(sqrd_dist(d_D_embed, dims_embed, i, j));
            attr = attr * attr * d_weight[l];
            for (int h = 0; h < dims_embed; h++) {
                d_grads[i * dims_embed + h] +=
                        attr * (d_D_embed[i * dims_embed + h] - d_D_embed[j * dims_embed + h]);
            }

            int g = curand(&d_random[id]) % n;//random int
            float rep = q(sqrd_dist(d_D_embed, dims_embed, i, g));
            rep = rep * rep * rep;
            for (int h = 0; h < dims_embed; h++) {
                d_grads[i * dims_embed + h] -=
                        rep * (d_D_embed[i * dims_embed + h] - d_D_embed[g * dims_embed + h]);

            }
        }
    }
}


//// for any k
//__global__
//void compute_grads_head_tail(float *d_grads, float *d_P, int n_edges, int n_vertices, int *d_heads, int *d_tails,
//                             float *d_D_embed, int dims_embed, float lr, curandState *d_random) {
//    int id = threadIdx.x + blockDim.x * blockIdx.x;
//
//    for (int i_edge = threadIdx.x + blockIdx.x * blockDim.x; i_edge < n_edges; i_edge += blockDim.x * gridDim.x) {
//        int i = d_heads[i_edge];
//        int j = d_tails[i_edge];
//        float attr = q(dist(d_D_embed, dims_embed, i, j));
//        attr = attr * attr * d_P[i_edge];
//        for (int h = 0; h < dims_embed; h++) {
//            d_grads[i * dims_embed + h] +=
//                    lr * attr * (d_D_embed[i * dims_embed + h] - d_D_embed[j * dims_embed + h]);
//        }
//
//        int g = curand(&d_random[id]) % n_vertices;//random int
//        float rep = q(dist(d_D_embed, dims_embed, i, g));
//        rep = rep * rep * rep;
//        for (int h = 0; h < dims_embed; h++) {
//            d_grads[i * dims_embed + h] -=
//                    lr * rep * (d_D_embed[i * dims_embed + h] - d_D_embed[g * dims_embed + h]);
//
//        }
//    }
//}

__global__
void apply_grads(float *d_D_embed, float *d_grads, int n, int dims_embed, float lr) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int h = 0; h < dims_embed; h++) {
            d_D_embed[i * dims_embed + h] += lr * d_grads[i * dims_embed + h];
        }
    }
}

float get_lr(float initial_lr, int i_epoch, int n_epochs) {
    return initial_lr * (1.0 - (((float) i_epoch) / ((float) n_epochs)));
}

//void gpu_umap_old(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P, int epochs,
//                  float init_lr, int neg_samples) {
//
//    //allocated and copy memory to the gpu
//    float *d_D_embed = copy_H_to_D(h_D_embed, n * dims_embed);
//    int *d_N = copy_H_to_D(h_N, n * k);
//    float *d_P = copy_H_to_D(h_P, n * k);
//    float *d_grads = gpu_malloc_float_zero(n * dims_embed);
//
//    //random
//    int number_of_threads = min(n, 32768);
//    int number_of_blocks = number_of_threads / BLOCK_SIZE;
//    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;
//    curandState *d_random;
//    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
//    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);
//
//    for (int epoch = 0; epoch < epochs; epoch++) {
//        float lr = get_lr(init_lr, epoch, epochs);
//        cudaMemset(d_grads, 0, n * dims_embed * sizeof(float));
//        compute_grads << < number_of_blocks, BLOCK_SIZE >> >
//        (d_grads, d_P, n, d_N, k, d_D_embed, dims_embed, lr, neg_samples, d_random);
//        apply_grads << < number_of_blocks, BLOCK_SIZE >> > (d_D_embed, d_grads, n, dims_embed);
//    }
//
//    //copy back and delete
//    cudaMemcpy(h_D_embed, d_D_embed, n * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(d_D_embed);
//    cudaFree(d_N);
//    cudaFree(d_P);
//    cudaFree(d_grads);
//    cudaFree(d_random);
//}


//void
//gpu_umap_head_tail(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_heads, int *h_tails,
//                   float *h_P, int epochs, float init_lr) {
//
//    //allocated and copy memory to the gpu
//    float *d_D_embed = copy_H_to_D(h_D_embed, n_vertices * dims_embed);
//    int *d_heads = copy_H_to_D(h_heads, n_edges);
//    int *d_tails = copy_H_to_D(h_tails, n_edges);
//    float *d_P = copy_H_to_D(h_P, n_edges);
//    float *d_grads = gpu_malloc_float_zero(n_vertices * dims_embed);
//
//    //random
//    int number_of_threads = min(n_vertices, 32768);
//    int number_of_blocks = number_of_threads / BLOCK_SIZE;
//    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;
//    curandState *d_random;
//    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
//    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);
//
//    for (int epoch = 0; epoch < epochs; epoch++) {
//        float lr = get_lr(init_lr, epoch, epochs);
//        cudaMemset(d_grads, 0, n_vertices * dims_embed * sizeof(float));
//        compute_grads_head_tail << < number_of_blocks, BLOCK_SIZE >> >
//        (d_grads, d_P, n_edges, n_vertices, d_heads, d_tails, d_D_embed, dims_embed, lr, d_random);
//        apply_grads << < number_of_blocks, BLOCK_SIZE >> > (d_D_embed, d_grads, n_vertices, dims_embed);
//    }
//
//    //copy back and delete
//    cudaMemcpy(h_D_embed, d_D_embed, n_vertices * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(d_D_embed);
//    cudaFree(d_heads);
//    cudaFree(d_tails);
//    cudaFree(d_P);
//    cudaFree(d_grads);
//    cudaFree(d_random);
//}

__global__
void convert(int *d_dst_int, long *d_src_long, int n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        d_dst_int[i] = (int) d_src_long[i];
    }
}

void
gpu_umap_2(int n, float *h_D_embed, int dims_embed, int *h_N, long *h_neighbor_counts, int k, float *h_P, int epochs,
           float init_lr, int neg_samples) {

    //allocated and copy memory to the gpu
    float *d_D_embed = copy_H_to_D(h_D_embed, n * dims_embed);
    int *d_N = copy_H_to_D(h_N, n * k);
    long *d_neighbor_counts_long = copy_H_to_D(h_neighbor_counts, n);
    int *d_neighbor_counts = gpu_malloc_int(n);
    int *d_neighbor_ends = gpu_malloc_int_zero(n);
    float *d_P = copy_H_to_D(h_P, n * k);
    float *d_grads = gpu_malloc_float_zero(n * dims_embed);


    //random
    int number_of_threads = min(n, 32768);
    int number_of_blocks = number_of_threads / BLOCK_SIZE;
    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;
    curandState *d_random;
    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);

    convert<<<number_of_blocks, BLOCK_SIZE>>>(d_neighbor_counts, d_neighbor_counts_long, n);
    inclusive_scan(d_neighbor_counts, d_neighbor_ends, n);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float lr = get_lr(init_lr, epoch, epochs);
        cudaMemset(d_grads, 0, n * dims_embed * sizeof(float));
        compute_grads << < number_of_blocks, BLOCK_SIZE >> >
        (d_grads, d_P, n, d_N, d_neighbor_ends, d_D_embed, dims_embed, d_random);
        apply_grads << < number_of_blocks, BLOCK_SIZE >> > (d_D_embed, d_grads, n, dims_embed, lr);
    }

    //copy back and delete
    cudaMemcpy(h_D_embed, d_D_embed, n * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_D_embed);
    cudaFree(d_N);
    cudaFree(d_neighbor_counts);
    cudaFree(d_P);
    cudaFree(d_grads);
    cudaFree(d_random);
}

void gpu_umap(
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
) {
    int k = n_edges / n_vertices;
    gpu_umap_2(n_vertices, head_embedding, dim, head, neighbor_counts, n_edges / n_vertices, weights, n_epochs,
               initial_lr, 1);
//    printf("head: ");
//    for (int i = 0; i < 30; i++) {
//        printf("%d ", head[i]);
//    }
//    printf("\n");
//    printf("tail: ");
//    for (int i = 0; i < 30; i++) {
//        printf("%d ", tail[i]);
//    }
//    printf("\n");
}