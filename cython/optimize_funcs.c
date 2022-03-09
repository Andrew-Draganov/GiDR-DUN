#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cython_utils.c"

static float get_kernels(
        float *attr_forces,
        float *rep_forces,
        float *attr_vecs,
        float *rep_vecs,
        int* head,
        int* tail,
        float* head_embedding,
        float* tail_embedding,
        float* weights,
        int normalized,
        int n_vertices,
        int n_edges,
        int i_epoch,
        int dim,
        float a,
        float b,
        float average_weight
){
    int edge, j, k, d;
    float dist_squared, weight_scalar;
    float *y1;
    float *y2;
    float *rep_func_outputs;

    float Z = 0;
    y1 = (float*) malloc(sizeof(float) * dim);
    y2 = (float*) malloc(sizeof(float) * dim);
    rep_func_outputs = (float*) malloc(sizeof(float) * 2);
    for(edge=0; edge<n_edges; edge++){
        j = head[edge];
        k = tail[edge];
        for(d=0; d<dim; d++){
            y1[d] = head_embedding[j * dim + d];
            y2[d] = tail_embedding[k * dim + d];
            attr_vecs[edge * dim + d] = y1[d] - y2[d];
        }
        dist_squared = sq_euc_dist(y1, y2, dim);

        // t-SNE early exaggeration
        if(i_epoch < 100)
            weight_scalar = 4;
        else
            weight_scalar = 1;

        attr_forces[edge] = attractive_force_func(
                normalized,
                dist_squared,
                a,
                b,
                weights[edge] * weight_scalar
        );

        k = rand() % n_vertices;
        for(d=0; d<dim; d++){
            y2[d] = tail_embedding[k * dim + d];
            rep_vecs[edge * dim + d] = y1[d] - y2[d];
        }
        dist_squared = sq_euc_dist(y1, y2, dim);
        repulsive_force_func(
                rep_func_outputs,
                normalized,
                dist_squared,
                a,
                b,
                1.0,
                average_weight
        );
        rep_forces[edge] = rep_func_outputs[0];
        Z += rep_func_outputs[1];
    }

    free(rep_func_outputs);
    free(y1);
    free(y2);

    return Z;
}

static void gather_gradients(
        float *attr_grads,
        float *rep_grads,
        int* head,
        int* tail,
        float* attr_forces,
        float* rep_forces,
        float* attr_vecs,
        float* rep_vecs,
        int sym_attraction,
        int n_vertices,
        int n_edges,
        int dim,
        float Z
){
    int j, k, d, edge;
    float grad_d;

    for(edge=0; edge<n_edges; edge++){
        j = head[edge];
        for(d=0; d<dim; d++){
            grad_d = clip(attr_forces[edge] * attr_vecs[edge * dim + d], -4, 4);
            attr_grads[j * dim + d] += grad_d;
            if(sym_attraction == 1){
                k = tail[edge];
                attr_grads[k * dim + d] -= grad_d;
            }
        }

        for(d=0; d<dim; d++){
            grad_d = clip(rep_forces[edge] * rep_vecs[edge * dim + d], -4, 4);
            rep_grads[j * dim + d] += grad_d / Z;
        }
    }
}