//
// Created by mrjak on 27-04-2021.
//

#ifndef GPU_PROCLUS_MEM_UTIL_H
#define GPU_PROCLUS_MEM_UTIL_H

template<typename T>
T *array_1d(int n);

template<typename T>
T **array_2d(int n, int m);

template<typename T>
T **zeros_2d(int n, int m);

template<typename T>
T *zeros_1d(int n);

int *fill_with_indices(int n);

template<typename T>
void free(T **X, int n);

template<typename T>
void free(T *X);

int get_allocated_count();

#endif //GPU_PROCLUS_MEM_UTIL_H
