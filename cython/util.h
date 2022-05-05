#ifndef PROCLUS_GPU_UTIL_H
#define PROCLUS_GPU_UTIL_H

#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>
#include <cstdlib>

int argmax_1d(float *values, int n);

template<typename T>
extern int argmin_1d(T *values, int n);

std::pair<int, int> *argmin_2d(float **values, int n, int m);

void index_wise_minimum(float *values_1, float *values_2, int n);

void index_wise_minimum_parallel(float *values_1, float *values_2, int n);

float mean_1d(float *values, int n);

bool all_close_1d(float *values_1, float *values_2, int n);

bool all_close_2d(float **values_1, float **values_2, int n, int m);

bool close(float value_1, float value_2);

int *shuffle(int *indices, int n);

int *random_sample(int *indices, int k, int n);

int *not_random_sample(int *in, int *state, int state_length, int k, int n);

float **gather_2d(float **S, int *indices, int k, int d);

void print_debug(char *str, bool debug);

void print_array(float *x, int n);

void print_array(int *x, int n);

void print_array(int *x, int *idx, int n);

void print_array(bool *x, int n);

void print_array(int **X, int n, int m);

void print_array(float **X, int n, int m);

void print_array(bool **X, int n, int m);

#endif //PROCLUS_GPU_UTIL_H
