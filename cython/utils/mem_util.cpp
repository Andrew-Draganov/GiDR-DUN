//
// Created by mrjak on 27-04-2021.
//

#include "mem_util.h"

int total_mem_allocations = 0;

template<typename T>
T *array_1d(int n) {
    T *S = new T[n];
    total_mem_allocations++;
    return S;
}

template int *array_1d<int>(int n);

template bool *array_1d<bool>(int n);

template float *array_1d<float>(int n);

template<typename T>
T **array_2d(int n, int m) {
    T **S = new T *[n];
    total_mem_allocations++;
    for (int i = 0; i < n; i++) {
        T *S_i = new T[m];
        total_mem_allocations++;
        S[i] = S_i;
    }
    return S;
}

template int **array_2d<int>(int n, int m);

template bool **array_2d<bool>(int n, int m);

template float **array_2d<float>(int n, int m);

template<typename T>
T **zeros_2d(int n, int m) {
    T **S = new T *[n];
    total_mem_allocations++;
    for (int i = 0; i < n; i++) {
        S[i] = new T[m];
        total_mem_allocations++;
        for (int j = 0; j < m; j++) {
            S[i][j] = 0;
        }
    }
    return S;
}

template int **zeros_2d<int>(int n, int m);

template bool **zeros_2d<bool>(int n, int m);

template float **zeros_2d<float>(int n, int m);

template<typename T>
T *zeros_1d(int n) {
    T *S = new T[n];
    total_mem_allocations++;
    for (int i = 0; i < n; i++) {
        S[i] = 0;
    }
    return S;
}

template int *zeros_1d<int>(int n);

template bool *zeros_1d<bool>(int n);

template float *zeros_1d<float>(int n);

int *fill_with_indices(int n) {
    int *h_S = new int[n];
    total_mem_allocations++;
    for (int p = 0; p < n; p++) {
        h_S[p] = p;
    }
    return h_S;
}

template<typename T>
void free(T *X) {
    delete X;
    total_mem_allocations--;
}

template void free<int>(int *X);

template void free<bool>(bool *X);

template void free<float>(float *X);

template<typename T>
void free(T **X, int n) {
    for (int i = 0; i < n; i++) {
        delete X[i];
        total_mem_allocations--;
    }
    delete X;
    total_mem_allocations--;
}

template void free<int>(int **X, int n);

template void free<bool>(bool **X, int n);

template void free<float>(float **X, int n);

int get_allocated_count() {
    return total_mem_allocations;
}