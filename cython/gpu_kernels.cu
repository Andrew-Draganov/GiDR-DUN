#import <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#import "gpu_kernels.h"


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
float dist(float *d_D, int dims, int i, int j) {
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

// for any k
__global__
void compute_grads(float *d_grads, float *d_P, int n, int *d_N, int k, float *d_D_embed,
                   int dims_embed, float lr, int neg_samples, curandState *d_random) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int l = 0; l < k; l++) {
            int j = d_N[i * k + l];
            float attr = q(dist(d_D_embed, dims_embed, i, j));
            attr = attr * attr * d_P[i * k + l];
            for (int h = 0; h < dims_embed; h++) {
                d_grads[i * dims_embed + h] +=
                        lr * attr * (d_D_embed[i * dims_embed + h] - d_D_embed[j * dims_embed + h]);
            }

            int g = curand(&d_random[id]) % n;//random int
            float rep = q(dist(d_D_embed, dims_embed, i, g));
            rep = rep * rep * rep;
            for (int h = 0; h < dims_embed; h++) {
                d_grads[i * dims_embed + h] -=
                        lr * rep * (d_D_embed[i * dims_embed + h] - d_D_embed[g * dims_embed + h]);

            }
        }
    }
}


// for any k
__global__
void compute_grads_head_tail(float *d_grads, float *d_P, int n_edges, int n_vertices, int *d_heads, int *d_tails,
                             float *d_D_embed, int dims_embed, float lr, curandState *d_random) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i_edge = threadIdx.x + blockIdx.x * blockDim.x; i_edge < n_edges; i_edge += blockDim.x * gridDim.x) {
        int i = d_heads[i_edge];
        int j = d_tails[i_edge];
        float attr = q(dist(d_D_embed, dims_embed, i, j));
        attr = attr * attr * d_P[i_edge];
        for (int h = 0; h < dims_embed; h++) {
            d_grads[i * dims_embed + h] +=
                    lr * attr * (d_D_embed[i * dims_embed + h] - d_D_embed[j * dims_embed + h]);
        }

        int g = curand(&d_random[id]) % n_vertices;//random int
        float rep = q(dist(d_D_embed, dims_embed, i, g));
        rep = rep * rep * rep;
        for (int h = 0; h < dims_embed; h++) {
            d_grads[i * dims_embed + h] -=
                    lr * rep * (d_D_embed[i * dims_embed + h] - d_D_embed[g * dims_embed + h]);

        }
    }
}

__global__
void apply_grads(float *d_D_embed, float *d_grads, int n, int dims_embed) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int h = 0; h < dims_embed; h++) {
            d_D_embed[i * dims_embed + h] += d_grads[i * dims_embed + h];
        }
    }
}

int *gpu_malloc_int(int n) {
    if (n <= 0)
        return nullptr;
    int *tmp;
    cudaMalloc(&tmp, n * sizeof(int));
    return tmp;
}


float *gpu_malloc_float(int n) {
    if (n <= 0)
        return nullptr;
    float *tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    return tmp;
}

float *gpu_malloc_float_zero(int n) {
    if (n <= 0)
        return nullptr;
    float *tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    cudaMemset(tmp, 0, n * sizeof(float));
    return tmp;
}

int *copy_H_to_D(int *h_array, int n) {
    int *d_array = gpu_malloc_int(n);
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    return d_array;
}

float *copy_H_to_D(float *h_array, int n) {
    float *d_array = gpu_malloc_float(n);
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    return d_array;
}

float get_lr(float initial_lr, int i_epoch, int n_epochs) {
    return initial_lr * (1.0 - (((float) i_epoch) / ((float) n_epochs)));
}

void gpu_umap_old(int n, float *h_D_embed, int dims_embed, int *h_N, int k, float *h_P, int epochs,
              float init_lr, int neg_samples) {

    //allocated and copy memory to the gpu
    float *d_D_embed = copy_H_to_D(h_D_embed, n * dims_embed);
    int *d_N = copy_H_to_D(h_N, n * k);
    float *d_P = copy_H_to_D(h_P, n * k);
    float *d_grads = gpu_malloc_float_zero(n * dims_embed);

    //random
    int number_of_threads = min(n, 32768);
    int number_of_blocks = number_of_threads / BLOCK_SIZE;
    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;
    curandState *d_random;
    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float lr = get_lr(init_lr, epoch, epochs);
        cudaMemset(d_grads, 0, n * dims_embed * sizeof(float));
        compute_grads << < number_of_blocks, BLOCK_SIZE >> >
        (d_grads, d_P, n, d_N, k, d_D_embed, dims_embed, lr, neg_samples, d_random);
        apply_grads << < number_of_blocks, BLOCK_SIZE >> > (d_D_embed, d_grads, n, dims_embed);
    }

    //copy back and delete
    cudaMemcpy(h_D_embed, d_D_embed, n * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_D_embed);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_grads);
    cudaFree(d_random);
}


void
gpu_umap_head_tail(int n_edges, int n_vertices, float *h_D_embed, int dims_embed, int *h_heads, int *h_tails,
                   float *h_P, int epochs, float init_lr) {

    //allocated and copy memory to the gpu
    float *d_D_embed = copy_H_to_D(h_D_embed, n_vertices * dims_embed);
    int *d_heads = copy_H_to_D(h_heads, n_edges);
    int *d_tails = copy_H_to_D(h_tails, n_edges);
    float *d_P = copy_H_to_D(h_P, n_edges);
    float *d_grads = gpu_malloc_float_zero(n_vertices * dims_embed);

    //random
    int number_of_threads = min(n_vertices, 32768);
    int number_of_blocks = number_of_threads / BLOCK_SIZE;
    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;
    curandState *d_random;
    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float lr = get_lr(init_lr, epoch, epochs);
        cudaMemset(d_grads, 0, n_vertices * dims_embed * sizeof(float));
        compute_grads_head_tail << < number_of_blocks, BLOCK_SIZE >> >
        (d_grads, d_P, n_edges, n_vertices, d_heads, d_tails, d_D_embed, dims_embed, lr, d_random);
        apply_grads << < number_of_blocks, BLOCK_SIZE >> > (d_D_embed, d_grads, n_vertices, dims_embed);
    }

    //copy back and delete
    cudaMemcpy(h_D_embed, d_D_embed, n_vertices * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_D_embed);
    cudaFree(d_heads);
    cudaFree(d_tails);
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
//    gpu_umap_old(n_vertices, head_embedding, dim, tail,n_edges/n_vertices, weights, int epochs,
//            initial_lr, int neg_samples);
}