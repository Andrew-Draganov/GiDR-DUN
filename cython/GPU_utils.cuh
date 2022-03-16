//
// Created by mrjak on 20-07-2021.
//

#ifndef GPU_SYNC_GPU_UTILS_H
#define GPU_SYNC_GPU_UTILS_H

void inclusive_scan(int *source, int *result, int n);

void inclusive_scan(unsigned int *source, unsigned int *result, int n);

int *gpu_malloc_int(int n);

long *gpu_malloc_long(int n);

unsigned int *gpu_malloc_unsigned_int(int n);

float *gpu_malloc_float(int n);

bool *gpu_malloc_bool(int n);

int *gpu_malloc_int_zero(int n);

float *gpu_malloc_float_zero(int n);

bool *gpu_malloc_bool_false(int n);

void copy_D_to_H(int *h_out, int *d_in, int n);

void copy_D_to_H(unsigned int *h_out, unsigned int *d_in, int n);

void copy_D_to_H(float *h_out, float *d_in, int n);

void copy_D_to_H(bool *h_out, bool *d_in, int n);

int *copy_D_to_H(int *d_array, int n);

float *copy_D_to_H(float *d_array, int n);

bool *copy_D_to_H(bool *d_array, int n);

int *copy_H_to_D(int *h_array, int n);

long *copy_H_to_D(long *h_array, int n);

float *copy_H_to_D(float *h_array, int n);

bool *copy_H_to_D(bool *h_array, int n);

float *copy_D_to_D(float *d_array1, int n);

void copy_H_to_D(int *d_out, int *h_in, int n);

void copy_H_to_D(float *d_out, float *h_in, int n);

void copy_H_to_D(bool *d_out, bool *h_in, int n);

void copy_D_to_D(int *d_out, int *d_in, int n);

void copy_D_to_D(float *d_out, float *d_in, int n);

void copy_D_to_D(bool *d_out, bool *d_in, int n);

int copy_last_D_to_H(int *d_array, int n);

int copy_last_D_to_H(unsigned int *d_array, int n);

float copy_last_D_to_H(float *d_array, int n);

void gpu_set_all_zero(int *d_var, int n);

void gpu_set_all_zero(unsigned int *d_var, int n);

void gpu_set_all_zero(float *d_var, int n);

void gpu_set_all(int *d_var, int size, int value);

void gpu_set_all(float *d_var, int size, float value);

void print_array_gpu(int *x, int n);

void print_array_nonzero_gpu(int *x, int n);

void print_array_gpu(float *x, int n);

void print_array_gpu(bool *x, int n);

void print_array_gpu(float *d_X, int n, int m);

void print_array_gpu(int *d_X, int n, int m);

#endif //GPU_SYNC_GPU_UTILS_H
