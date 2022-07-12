#import <stdio.h>
#include "gpu_graph_cuda.cuh"
#include "../utils/util.h"
#include "../utils/gpu_utils.cuh"
#import <limits>


#define BLOCK_SIZE 512
#define SMOOTH_K_TOLERANCE 1e-5
#define MIN_K_DIST_SCALE 1e-3

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

void cuda() {
    printf("hello from the cuda\n");
}

__global__
void kernel_rhos(float *d_rhos,
                 float *d_knn_dists,
                 int n_points,
                 int n_neighbors) {

    extern __shared__ float s[];

    float *non_zero_dists = &s[threadIdx.x * n_neighbors];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_points; i += blockDim.x * gridDim.x) {

        float *neighbor_distances = &d_knn_dists[i * n_neighbors];

        //non_zero_dists = ith_distances[ith_distances > 0.0]
        int number_of_non_zeros = 0;
        for (int j = 0; j < n_neighbors; j++) {
            if (neighbor_distances[j] > 0.0) {
                non_zero_dists[number_of_non_zeros] = neighbor_distances[j];
                number_of_non_zeros++;
            }
        }
        if (number_of_non_zeros >= 1.0) {
            d_rhos[i] = non_zero_dists[0];
        } else if (number_of_non_zeros > 0) {
            //rho[i] = np.max(non_zero_dists)
            d_rhos[i] = non_zero_dists[0];
            for (int j = 1; j < number_of_non_zeros; j++) {
                if (non_zero_dists[j] > d_rhos[i]) {
                    d_rhos[i] = non_zero_dists[j];
                }
            }
        }
    }
}

__global__
void kernel_sigmas(
        float *d_rhos,
        float *d_sigmas,
        float *d_knn_dists,
        int n_points,
        int n_neighbors,
        int n_itr,
        int pseudo_distance,
        float target,
        float mean_distances,
        float float_max
) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_points; i += blockDim.x * gridDim.x) {
        float lo = 0.;
        float hi = float_max;
        float mid = 1.;

        for (int itr = 0; itr < n_itr; itr++) {
            float psum = 0.0;
            for (int j = 1; j < n_neighbors; j++) { // i assume that j=0, is the same point as i?
                float d = d_knn_dists[i * n_neighbors + j];
                if (pseudo_distance) {
                    d -= d_rhos[i];
                }
                if (d > 0.) {
                    psum += exp(-(d / mid));
                } else {
                    psum += 1.0;
                }
            }

            if (abs(psum - target) < SMOOTH_K_TOLERANCE) {
                break;
            }

            if (psum > target) {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if (hi == float_max) {
                    mid *= 2;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
            d_sigmas[i] = mid;
        }
        if (d_rhos[i] > 0.) {
            float mean_ith_distances = 0.;
            for (int j = 0; j < n_neighbors; j++) {
                mean_ith_distances += d_knn_dists[i * n_neighbors + j];
            }
            mean_ith_distances /= n_neighbors;
            if (d_sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances) {
                d_sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances;
            } else {

                if (d_sigmas[i] < MIN_K_DIST_SCALE * mean_distances) {
                    d_sigmas[i] = MIN_K_DIST_SCALE * mean_distances;
                }
            }
        }
    }
}


__global__
void kernel_compute_P(
        int *d_rows,
        int *d_cols,
        float *d_vals,
        float *d_dists,
        int *d_knn_indices,
        float *d_knn_dists,
        float *d_rhos,
        float *d_sigmas,
        int n_points,
        int n_neighbors,
        int pseudo_distance
) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_points; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_neighbors; j++) {
            if (d_knn_indices[i * n_neighbors + j] == -1) {
                continue;
            }
            float val;
            if (d_knn_indices[i * n_neighbors + j] == i) {
                val = 0.0;
            } else if (d_knn_dists[i * n_neighbors + j] - d_rhos[i] <= 0.0 || d_sigmas[i] == 0.0) {
                val = 1.0;
            } else {
                if (pseudo_distance) {
                    val = exp(-((d_knn_dists[i * n_neighbors + j] - d_rhos[i]) / (d_sigmas[i])));
                } else {
                    val = exp(-((d_knn_dists[i * n_neighbors + j]) / (d_sigmas[i])));
                }
            }
            d_rows[i * n_neighbors + j] = i;
            d_cols[i * n_neighbors + j] = d_knn_indices[i * n_neighbors + j];
            d_vals[i * n_neighbors + j] = val;
            d_dists[i * n_neighbors + j] = d_knn_dists[i * n_neighbors + j];
        }
    }
}

void compute_neighbor_graph_cuda(
        int *rows, //return val
        int *cols, //return val
        float *values, //return val
        float *dists, //return val
        float *sigmas, //return val
        float *rhos, //return val
        int *knn_indices,
        float *knn_dists,
        int n_points,
        int n_neighbors,
        int pseudo_distance
) {
    float bandwidth = 1.;
    float target = log2(n_neighbors) * bandwidth;
    float mean_distances = mean_1d(knn_dists, n_points * n_neighbors);

    float *d_knn_dists = copy_H_to_D(knn_dists, n_points * n_neighbors);
    int *d_knn_indices = copy_H_to_D(knn_indices, n_points * n_neighbors);
    float *d_rhos = gpu_malloc_float_zero(n_points);
    float *d_sigmas = gpu_malloc_float_zero(n_points);
    int *d_rows = gpu_malloc_int_zero(n_points * n_neighbors);
    int *d_cols = gpu_malloc_int_zero(n_points * n_neighbors);
    float *d_vals = gpu_malloc_float_zero(n_points * n_neighbors);
    float *d_dists = gpu_malloc_float_zero(n_points * n_neighbors);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //todo do we want to do more then one thing at a time?
    int number_of_blocks = n_points / BLOCK_SIZE;
    if (n_points % BLOCK_SIZE) number_of_blocks++;

    kernel_rhos<<<number_of_blocks, BLOCK_SIZE, n_neighbors * BLOCK_SIZE * sizeof(float)>>>(
            d_rhos,
            d_knn_dists,
            n_points,
            n_neighbors
    );

    kernel_sigmas<<<number_of_blocks, BLOCK_SIZE>>>(
            d_rhos,
            d_sigmas,
            d_knn_dists,
            n_points,
            n_neighbors,
            64,
            pseudo_distance,
            target,
            mean_distances,
            std::numeric_limits<float>::max()
    );

    kernel_compute_P<<<number_of_blocks, BLOCK_SIZE>>>(
            d_rows,
            d_cols,
            d_vals,
            d_dists,
            d_knn_indices,
            d_knn_dists,
            d_rhos,
            d_sigmas,
            n_points,
            n_neighbors,
            pseudo_distance
    );


    copy_D_to_H(rhos, d_rhos, n_points);
    copy_D_to_H(sigmas, d_sigmas, n_points);
    copy_D_to_H(rows, d_rows, n_points * n_neighbors);
    copy_D_to_H(cols, d_cols, n_points * n_neighbors);
    copy_D_to_H(values, d_vals, n_points * n_neighbors);
    copy_D_to_H(dists, d_dists, n_points * n_neighbors);

    cudaFree(d_knn_dists);
    cudaFree(d_knn_indices);
    cudaFree(d_rhos);
    cudaFree(d_sigmas);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_dists);
}
