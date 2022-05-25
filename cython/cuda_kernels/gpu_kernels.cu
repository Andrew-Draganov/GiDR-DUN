#import <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#import "gpu_kernels.h"
#import "../utils/gpu_utils.cuh"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

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


#define BLOCK_SIZE 256//1024

/// https://github.com/rapidsai/cuml/blob/branch-22.04/cpp/src/umap/runner.cuh

__device__
float sqrd_dist(const float *__restrict__ d_D, const int dims, const int i, const int j) {
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


__device__
int get_start(const int *d_ends, int i) {
    return i == 0 ? 0 : d_ends[i - 1];
}

__device__
int get_end(const int *d_ends, int i) {
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
float kernel_function(float dist_squared, float a, float b) {
    if (b <= 1)
        return 1 / (1 + a * fast_pow(dist_squared, b));
    return fast_pow(dist_squared, b - 1) / (1 + a * fast_pow(dist_squared, b));
}

__device__
float umap_attr_scalar(float dist_squared, float a, float b) {
    float grad_scalar = 0.0;
    grad_scalar = 2.0 * a * b * fast_pow(dist_squared, b - 1.0);
    grad_scalar /= a * fast_pow(dist_squared, b) + 1.0;
    return grad_scalar;
}

__device__
float frob_attr_force(int normalized, float p, float q) {
    if (normalized) {
        // FIXME - is it faster to get q^2 and then use that for q^3?
        // FIXME - took out a Z scalar from this
        return p * (q * q + 2 * pow(q, 3));
    }
    return p * q * q;
}

__device__
float kl_attr_force(float p, float q) {
    return p * q;
}

__device__
float attractive_force_func(
        int frob,
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
) {

    float q;
    if (normalized || frob)
        q = kernel_function(dist_squared, a, b);
    else
        q = umap_attr_scalar(dist_squared, a, b);

    if (frob)
        return frob_attr_force(normalized, edge_weight, q);
    else
        return kl_attr_force(edge_weight, q);


}
__device__
float norm_rep_force(
        float *d_Z,
        int i_thread,
        float dist_squared,
        float a,
        float b,
        float cell_size
) {
    float kernel, q_ij, repulsive_force;

    kernel = kernel_function(dist_squared, a, b);

    q_ij = cell_size * kernel; // Collect the q_ij's contributions into Z
    d_Z[i_thread] += q_ij;

    repulsive_force = cell_size * kernel * kernel;
    return repulsive_force;
}

__device__
float umap_rep_scalar(float dist_squared, float a, float b) {
    float phi_ijZ = 0.0;
    phi_ijZ = 2.0 * b;
    phi_ijZ /= (0.001 + dist_squared) * (a * fast_pow(dist_squared, b) + 1);
    return phi_ijZ;
}

__device__
float frob_rep_force(int normalized, float q) {
    if (normalized)
        return pow(q, 3) + 2 * pow(q, 4);
    return pow(q, 3);
}

__device__
float kl_rep_force(int normalized, float q, float avg_weight) {
    if (normalized)
        return q * q;
    return q * (1 - avg_weight);
}

__device__
float repulsive_force_func(
        float *d_Z,
        int i_thread,
        int frob,
        int normalized,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight
) {
    float q, result;
    if (normalized || frob)
        q = kernel_function(dist_squared, a, b);
    else
        q = umap_rep_scalar(dist_squared, a, b);

    if (frob)
        result = frob_rep_force(normalized, q);
    else
        result = kl_rep_force(normalized, q, average_weight);
    result *= cell_size;

    if (normalized)
        d_Z[i_thread] += q * cell_size;

    return result;
}

__global__
void
compute_grads_full_shared_mem_N(const int frob, const int normalized, float *__restrict__ d_rep_grads,
                                float *__restrict__ d_attr_grads,
                                const float *__restrict__ d_weights,
                                const int n,
                                const int *__restrict__ d_N,
                                const int *__restrict__ d_neighbor_ends,
                                const float *__restrict__ d_D_embed,
                                float *__restrict__ d_Z, const float a, const float b, const int dims_embed,
                                curandState *__restrict__ d_random, const float sym_attraction,
                                const float weight_scalar,
                                const float average_weight, const int k, const int negative_sample_rate) {

    extern __shared__ float s_rep_grads[];
    float *s_attr_grads = &s_rep_grads[blockDim.x * dims_embed];

    int i_thread = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {

        for (int h = 0; h < dims_embed; h++) {
            s_rep_grads[tid * dims_embed + h] = 0.;
            s_attr_grads[tid * dims_embed + h] = 0.;
        }

        for (int i_edge = get_start(d_neighbor_ends, i_point); i_edge < get_end(d_neighbor_ends, i_point); i_edge++) {
            int j_point = d_N[i_edge];

            float dist_squared = sqrd_dist(d_D_embed, dims_embed, i_point, j_point);
            float attr = attractive_force_func(
                    frob,
                    normalized,
                    dist_squared,
                    a,
                    b,
                    d_weights[i_edge] * weight_scalar
            );
            for (int h = 0; h < dims_embed; h++) {

                float force = attr * (d_D_embed[i_point * dims_embed + h] - d_D_embed[j_point * dims_embed + h]);
                s_attr_grads[tid * dims_embed + h] -= force;
                atomicAdd(&d_attr_grads[j_point * dims_embed + h],
                          force * sym_attraction);
            }

            int g = curand(&d_random[i_thread]) % n;//random int
            dist_squared = sqrd_dist(d_D_embed, dims_embed, i_point, g);
            float rep = repulsive_force_func(
                    d_Z,
                    i_thread,
                    frob,
                    normalized,
                    dist_squared,
                    a,
                    b,
                    1.0,
                    average_weight
            );

            for (int h = 0; h < dims_embed; h++) {
                s_rep_grads[tid * dims_embed + h] +=
                        rep * (d_D_embed[i_point * dims_embed + h] - d_D_embed[g * dims_embed + h]);
            }


        }

        for (int h = 0; h < dims_embed; h++) {

            atomicAdd(&d_rep_grads[i_point * dims_embed + h], s_rep_grads[tid * dims_embed + h]);
            atomicAdd(&d_attr_grads[i_point * dims_embed + h], s_attr_grads[tid * dims_embed + h]);

        }
    }
}

__device__
float clip(float val, float lower, float upper) {
    return fmax(lower, fmin(val, upper));
}

__global__
void
apply_grads_full(float *d_Z, float *d_D_embed, float *d_rep_grads, float *d_attr_grads, float *d_all_grads,
                 float *d_gains,
                 int n, int dims_embed, float lr, float a, float b, float amplify_graps) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        for (int h = 0; h < dims_embed; h++) {
            int index = i * dims_embed + h;
            float grad = (d_rep_grads[index] / d_Z[0] + d_attr_grads[index]) * 4 * a * b;

            if (grad * d_all_grads[index] > 0.0)
                d_gains[index] += 0.2;
            else
                d_gains[index] *= 0.8;
            d_gains[index] = clip(d_gains[index], 0.01, 100);
            grad *= d_gains[index];

            d_all_grads[index] *= (float) amplify_graps * 0.9;
            d_all_grads[index] += grad * lr;

            d_D_embed[index] += d_all_grads[index];
        }
    }
}

float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_graps) {
    if (amplify_graps)
        return initial_lr;
    return initial_lr * (1.0 - (((float) i_epoch) / ((float) n_epochs)));
}

__global__
void convert(int *d_dst_int, long *d_src_long, int n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        d_dst_int[i] = (int) d_src_long[i];
    }
}

__global__
void reduced_sum_fix(float *d_out, float *d_in, int n) {
    extern __shared__ float s_tmp[];
    int i_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    s_tmp[tid] = 0;
    for (int s = 0; s < n; s += gridDim.x * blockDim.x) {
        if (i_thread + s < n) {
            s_tmp[tid] += d_in[i_thread + s];
        }
    }
    d_out[i_thread] = s_tmp[tid];
}

__global__
void reduced_sum(float *d_out, float *d_in, int n) {

    extern __shared__ float s_tmp[];

    int i_thread = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    int tid = threadIdx.x;

    int n_active_threads = blockDim.x;

    s_tmp[tid] = 0;
    if (i_thread < n)
        s_tmp[tid] += d_in[i_thread];
    if (i_thread + blockDim.x < n)
        s_tmp[tid] += d_in[i_thread + blockDim.x];
    __syncthreads();

    for (int n_active_threads = blockDim.x / 2; n_active_threads > 0; n_active_threads >>= 1) {
        if (tid < n_active_threads) {
            s_tmp[tid] += s_tmp[tid + n_active_threads];
            __syncthreads();
        }
    }

    if (tid == 0) d_out[blockIdx.x] = s_tmp[0];
}

float mean(float *h_x, int n) {
    float x = 0;
    for (int i = 0; i < n; i++) {
        x += h_x[i];
    }
    return x / n;
}

__global__
void compute_max(int *d_out, int *d_in, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        atomicMax(&d_out[0], d_in[i]);
    }
}

int gpu_max(int *d_in, int n) {
    int *d_out = gpu_malloc_int_zero(1);
    compute_max<<<32, BLOCK_SIZE>>>(d_out, d_in, n);
    int m = copy_last_D_to_H(d_out, 1);
    cudaFree(d_out);
    return m;
}

__global__
void pack_N(int *d_N_new, float *d_weights_new, int *d_N, float *d_weights, int *d_neighbor_ends, int n_vertices,
            int k) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n_vertices; i_point += blockDim.x * gridDim.x) {
        int loc = 0;
        for (int i_edge = get_start(d_neighbor_ends, i_point); i_edge < get_end(d_neighbor_ends, i_point); i_edge++) {
            d_N_new[i_point * k + loc] = d_N[i_edge];
            d_weights_new[i_point * k + loc] = d_weights[i_edge];
            loc++;
        }
    }
}


void gpu_umap_full_N(int normalized, // unused
                     int sym_attraction, // unused
                     int frob,
                     int amplify_graps, // unused
                     float *h_D_embed, //head_embedding,
                     float *h_D_embed_other, //tail_embedding,
                     int *h_N, //head,
                     int *tail, // im not using this
                     float *h_weights,//weights,
                     long *h_neighbor_counts, //neighbor_counts,
                     float *all_updates, // unused
                     float *gains, // unused
                     float a, // unused
                     float b, // unused
                     int dims_embed, //dim,
                     int n_vertices,
                     float init_lr,
                     int n_epochs,
                     int n_edges,
                     int negative_sample_rate
) {
    cudaDeviceSynchronize();
    int number_of_blocks_scalar = 32;//32 can be replace with something smaller then BLOCK_SIZE
    int number_of_threads_in_total = BLOCK_SIZE * 2 * number_of_blocks_scalar;

    //allocated and copy memory to the gpu
    float *d_D_embed = copy_H_to_D(h_D_embed, n_vertices * dims_embed);
    int *d_N = copy_H_to_D(h_N, n_edges);
    long *d_neighbor_counts_long = copy_H_to_D(h_neighbor_counts, n_vertices);
    int *d_neighbor_counts = gpu_malloc_int(n_vertices);
    int *d_neighbor_ends = gpu_malloc_int_zero(n_vertices);
    float *d_weights = copy_H_to_D(h_weights, n_edges);
    float *d_rep_grads = gpu_malloc_float(n_vertices * dims_embed);
    float *d_attr_grads = gpu_malloc_float(n_vertices * dims_embed);
    float *d_all_grads = gpu_malloc_float_zero(n_vertices * dims_embed);
    float *d_gains = gpu_malloc_float(n_vertices * dims_embed);
    gpu_set_all(d_gains, n_vertices * dims_embed, 1.);
    float *d_Z = gpu_malloc_float(number_of_threads_in_total);

    float *d_tmp_sum_1 = gpu_malloc_float(number_of_threads_in_total);
    float *d_tmp_sum_2 = gpu_malloc_float(number_of_blocks_scalar);


    int number_of_threads = min(n_vertices, number_of_threads_in_total);
    int number_of_blocks = number_of_threads / BLOCK_SIZE;
    if (number_of_threads % BLOCK_SIZE) number_of_blocks++;

    int number_of_blocks_half = (number_of_threads_in_total / 2) / BLOCK_SIZE;
    if ((number_of_threads_in_total / 2) % BLOCK_SIZE) number_of_blocks_half++;


    //random
    curandState *d_random;
    cudaMalloc((void **) &d_random, number_of_threads * sizeof(curandState));
    init_random << < number_of_blocks, BLOCK_SIZE >> > (d_random);

    convert<<<number_of_blocks, BLOCK_SIZE>>>(d_neighbor_counts, d_neighbor_counts_long, n_vertices);
    inclusive_scan(d_neighbor_counts, d_neighbor_ends, n_vertices);

    int k = gpu_max(d_neighbor_counts, n_vertices);

    reduced_sum_fix<<<number_of_blocks_half * 2, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>
            (d_tmp_sum_1, d_weights, n_edges);
    reduced_sum<<<number_of_blocks_half, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>
            (d_tmp_sum_2, d_tmp_sum_1, number_of_threads_in_total);
    reduced_sum<<<1, number_of_blocks_scalar, number_of_blocks_scalar * sizeof(float)>>>
            (d_tmp_sum_1, d_tmp_sum_2, number_of_blocks_scalar);
    float average_weight = copy_last_D_to_H(d_tmp_sum_1, 1) / n_edges;

//    printf("\n\nParams:\n");
//    printf("- average_weight: %f\n", average_weight);
//    printf("- amplify_graps: %d\n", amplify_graps);
//    printf("- sym_attraction: %d\n", sym_attraction);
//    printf("- normalized: %d\n", normalized);
//    printf("- n_edges: %d\n", n_edges);
//    printf("- negative_sample_rate: %d\n", negative_sample_rate);
//    printf("- a: %f\n", a);
//    printf("- b: %f\n", b);
//    printf("- number_of_blocks_scalar: %d\n", number_of_blocks_scalar);
//    printf("- number_of_blocks_half: %d\n", number_of_blocks_half);
//    printf("- number_of_blocks: %d\n", number_of_blocks);
//    printf("\n\n");

    gpu_set_all(d_tmp_sum_2, 1, 1.);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    for (int i_epoch = 0; i_epoch < n_epochs; i_epoch++) {
        float lr = get_lr(init_lr, i_epoch, n_epochs, amplify_graps);
        cudaMemset(d_rep_grads, 0, n_vertices * dims_embed * sizeof(float));
        cudaMemset(d_attr_grads, 0, n_vertices * dims_embed * sizeof(float));
        cudaMemset(d_Z, 0, number_of_threads_in_total * sizeof(float));
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        float weight_scalar;
        if (i_epoch < 100)
            weight_scalar = 4;
        else
            weight_scalar = 1;

        int number_of_blocks_n = n_vertices/BLOCK_SIZE;
        if (n_vertices % BLOCK_SIZE) number_of_blocks++;

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        compute_grads_full_shared_mem_N <<<
        number_of_blocks,
        BLOCK_SIZE,
        BLOCK_SIZE * dims_embed * 2 * sizeof(float)>>>
                (frob, normalized, d_rep_grads, d_attr_grads,
                 d_weights,
                 n_vertices,
                 d_N, d_neighbor_ends,
                 d_D_embed, d_Z,
                 a, b, dims_embed, d_random, sym_attraction, weight_scalar, average_weight, k, negative_sample_rate);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        float Z = 0.;
        if (normalized) {
            reduced_sum<<<number_of_blocks_half, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>
                    (d_tmp_sum_1, d_Z, number_of_threads_in_total);
            reduced_sum<<<1, number_of_blocks_scalar, number_of_blocks_scalar * sizeof(float)>>>
                    (d_tmp_sum_2, d_tmp_sum_1, number_of_blocks_scalar);
        }

        apply_grads_full << < number_of_blocks, BLOCK_SIZE>> >
        (d_tmp_sum_2, d_D_embed, d_rep_grads, d_attr_grads, d_all_grads, d_gains, n_vertices, dims_embed, lr, a, b, amplify_graps);


        if ((i_epoch + 1) % 50 == 0) {
            printf("Epoch %d/%d\n", i_epoch + 1, n_epochs);
        }

    }

    //copy back and delete
    cudaMemcpy(h_D_embed, d_D_embed, n_vertices * dims_embed * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_D_embed);
    cudaFree(d_N);
    cudaFree(d_neighbor_counts_long);
    cudaFree(d_neighbor_counts);
    cudaFree(d_neighbor_ends);
    cudaFree(d_weights);
    cudaFree(d_rep_grads);
    cudaFree(d_attr_grads);
    cudaFree(d_all_grads);
    cudaFree(d_gains);
    cudaFree(d_random);
    cudaFree(d_Z);
    cudaFree(d_tmp_sum_1);
    cudaFree(d_tmp_sum_2);
}


void gpu_umap(
        int normalized,
        int sym_attraction,
        int frob,
        int amplify_graps,
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
    int k = n_edges / n_vertices;
    gpu_umap_full_N(
            normalized, // unused
            sym_attraction, // unused
            frob,
            amplify_graps, // unused
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            neighbor_counts,
            all_updates, // unused
            gains, // unused
            a, // unused
            b, // unused
            dim,
            n_vertices,
            initial_lr,
            n_epochs,
            n_edges,
            negative_sample_rate
    );
}



__global__
void KNN(int *d_neighbors, float *d_distances, float *d_data, int n, int d, int k) {
    extern __shared__ float s_array[];
    float *s_distances = &s_array[2 * k * threadIdx.x];
    int *s_neighbors = (int *) &s_distances[k];
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {

        for (int i = 0; i < k; i++) {
            s_distances[i] = 0.;
        }

        for (int j_point = 0; j_point < n; j_point) {
            float distance = 0.;
            for (int i_dim = 0; i_dim < d; i_dim++) {
                float diff = d_data[i_point * d + i_dim] - d_data[j_point * d + i_dim];
                distance += diff * diff;
            }
            distance = sqrt(distance);
        }
    }
}

void GPU_KNN(int *h_neighbors, float *h_distances, float *h_data, int n, int d, int k) {


    ///test
    int *d_neighbors = gpu_malloc_int(n * k);
    float *d_distances = gpu_malloc_float(n * k);
    float *d_data = copy_H_to_D(h_data, n * d);
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)number_of_blocks++;
    KNN<<<number_of_blocks, BLOCK_SIZE>>>(d_neighbors, d_distances, d_data, n, d, k);
}