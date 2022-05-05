#include <cstdio>
#include "util.h"
#include "mem_util.h"
#include "omp.h"


#define DEBUG false

int argmax_1d(float *values, int n) {
    int max_idx = -1;
    float max_value = -10000;//todo something smaller
    //printf("min: %f\n", max_value);
    for (int i = 0; i < n; i++) {
        if (values[i] >= max_value) {
            max_value = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

template<typename T>
int argmin_1d(T *values, int n) {
    int min_idx = -1;
    T min_value = std::numeric_limits<T>::max();
    for (int i = 0; i < n; i++) {
        if (values[i] < min_value) {
            min_value = values[i];
            min_idx = i;
        }
    }
    return min_idx;
}

template int argmin_1d<int>(int *values, int n);

template int argmin_1d<float>(float *values, int n);

std::pair<int, int> *argmin_2d(float **values, int n, int m) {
    int min_x = -1;
    int min_y = -1;
    float min_value = std::numeric_limits<float>::max();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (values[i][j] < min_value) {
                min_value = values[i][j];
                min_x = i;
                min_y = j;
            }
        }
    }
    return new std::pair<int, int>(min_x, min_y);
}

void index_wise_minimum(float *values_1, float *values_2, int n) {
    for (int i = 0; i < n; i++) {
        values_1[i] = std::min(values_1[i], values_2[i]);
    }
}

void index_wise_minimum_parallel(float *values_1, float *values_2, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        values_1[i] = std::min(values_1[i], values_2[i]);
    }
}

float mean_1d(float *values, int n) {
    float sum = 0.;
    for (int i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum / (float) n;
}

bool all_close_1d(float *values_1, float *values_2, int n) {
    bool result = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(values_1[i] - values_2[i]) > 0.001)
            result = false;
    }
    return result;
}

bool all_close_2d(float **values_1, float **values_2, int n, int m) {
    bool result = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (std::abs(values_1[i][j] - values_2[i][j]) > 0.001)
                result = false;
        }
    }
    return result;
}

bool close(float value_1, float value_2) {
    return std::abs(value_1 - value_2) < 0.001;
}

int *shuffle(int *indices, int n) {

    //print_debug("shuffle - start\n", DEBUG);

    int *a = array_1d<int>(n);
    for (int i = 0; i < n; i++) {
        a[i] = indices[i];
    }

    for (int i = n - 1; i > 0; i--) {
        int j = std::rand() % (i + 1);
        int tmp_idx = a[i];
        a[i] = a[j];
        a[j] = tmp_idx;
    }

    //print_debug("shuffle - end\n", DEBUG);
    return a;
}

int *random_sample(int *indices, int k, int n) {
    for (int i = 0; i < k; i++) {
        int j = std::rand() % n;
        int tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
    }
    return indices;
}

int *not_random_sample(int *in, int *state, int state_length, int k, int n) {
    for (int i = 0; i < k; i++) {
        int j = state[0] % n;//i % state_length
        state[0] += 11;

        int tmp_idx = in[i];
        in[i] = in[j];
        in[j] = tmp_idx;
    }
    return in;
}

float **gather_2d(float **S, int *indices, int k, int d) {
    float **R = array_2d<float>(k, d);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            R[i][j] = S[indices[i]][j];
        }
    }
    return R;
}


//void print_debug(char *str, bool debug) {
//    if (debug)
//        printf(str);
//}

void print_array(float *x, int n) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%f ", (float) x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%f ", (float) x[i]);
        }
    }
    printf("\n");
}

void print_array(int *x, int n) {
    int left = 300;
    int right = 300;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", x[i]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", x[i]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", x[i]);
        }
    }
    printf("\n");
}

void print_array(int *x, int *idx, int n) {
    int left = 300;
    int right = 300;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%d ", x[idx[i]]);
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%d ", x[idx[i]]);
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%d ", x[idx[i]]);
        }
    }
    printf("\n");
}

void print_array(bool *x, int n) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
    } else {
        for (int i = 0; i < left; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
        printf(" ... ");
        for (int i = n - right; i < n; i++) {
            printf("%s ", x[i] ? "true" : "false");
        }
    }
    printf("\n");
}

void print_array(int **X, int n, int m) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            print_array(X[i], m);
        }
    } else {
        for (int i = 0; i < left; i++) {
            print_array(X[i], m);
        }
        printf(" ... \n");
        for (int i = n - right; i < n; i++) {
            print_array(X[i], m);
        }
    }
    printf("\n");
}

void print_array(float **X, int n, int m) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            print_array(X[i], m);
        }
    } else {
        for (int i = 0; i < left; i++) {
            print_array(X[i], m);
        }
        printf(" ... \n");
        for (int i = n - right; i < n; i++) {
            print_array(X[i], m);
        }
    }
    printf("\n");
}

void print_array(bool **X, int n, int m) {
    int left = 30;
    int right = 30;

    if (n <= left + right) {
        for (int i = 0; i < n; i++) {
            print_array(X[i], m);
        }
    } else {
        for (int i = 0; i < left; i++) {
            print_array(X[i], m);
        }
        printf(" ... \n");
        for (int i = n - right; i < n; i++) {
            print_array(X[i], m);
        }
    }
//    printf("\n");
}