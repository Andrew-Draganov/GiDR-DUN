#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cython_utils.c"

static void simple_single_epoch(
    int normalized, // unused
    int sym_attraction, // unused
    int momentum, // unused
    float* head_embedding,
    float* tail_embedding,
    int* head,
    int* tail,
    float* weights,
    long* neighbor_counts,
    float* all_updates, // unused
    float* gains, // unused
    float a, // unused
    float b, // unused
    int dim,
    int n_vertices,
    float lr,
    int i_epoch,
    int n_edges
){
    //
    // This is the simple version of the code for uniform_umap
    //
    int j, k, d, index;
    float dist_squared, weight_scalar;
    float *y1;
    float *y2;
    float *rep_func_outputs;
    float *grads;
    float force;

    float Z = 0;
    y1 = (float*) malloc(sizeof(float) * dim);
    y2 = (float*) malloc(sizeof(float) * dim);
    rep_func_outputs = (float*) malloc(sizeof(float) * 2);
    grads = (float*) malloc(sizeof(float) * n_vertices * dim);
    for(int v=0; v<n_vertices; v++){
        for(d=0; d<dim; d++){
            grads[v * dim + d] = 0;
        }
    }
    int edge = 0;
    for(int v=0; v<n_vertices; v++){
        for(int nbr=0; nbr<neighbor_counts[v]; nbr++){
            j = head[edge];
            for(d=0; d<dim; d++){
                y1[d] = tail_embedding[v * dim + d];
                y2[d] = head_embedding[j * dim + d];
            }
            dist_squared = sq_euc_dist(y1, y2, dim);

            force = attractive_force_func(
                    normalized,
                    dist_squared,
                    a,
                    b,
                    weights[edge]
            );
            for(d=0; d<dim; d++){
                grads[v * dim + d] -= force * (y1[d] - y2[d]);
            }

            k = rand() % n_vertices;
            for(d=0; d<dim; d++){
                y2[d] = head_embedding[k * dim + d];
            }
            dist_squared = sq_euc_dist(y1, y2, dim);
            repulsive_force_func(
                    rep_func_outputs,
                    normalized,
                    dist_squared,
                    a,
                    b,
                    1.0,
                    0.3 // FIXME -- make avg_weight
            );
            force = rep_func_outputs[0];
            for(d=0; d<dim; d++){
                grads[v * dim + d] += force * (y1[d] - y2[d]);
            }

            edge++;
        }
    }

    for(int v=0; v<n_vertices; v++){
        for(d=0; d<dim; d++){
            head_embedding[v * dim + d] += lr * grads[v * dim + d];
        }
    }

    free(grads);
    free(rep_func_outputs);
    free(y1);
    free(y2);
}


static void full_single_epoch(
    int normalized,
    int sym_attraction,
    int momentum,
    float* head_embedding,
    float* tail_embedding,
    int* head,
    int* tail,
    float* weights,
    long* neighbor_counts,
    float* all_updates,
    float* gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    int i_epoch,
    int n_edges
){
    //
    // This is the full version of the code for uniform_umap
    // NOTE: this is still not the full version that uses the frobenius norm
    //       but is very, very close
    //
    int j, k, d, index;
    float dist_squared, weight_scalar, grad_d, force;
    float *y1;
    float *y2;
    float *rep_func_outputs;
    float *attr_grads;
    float *rep_grads;
    float Z = 0;

    y1 = (float*) malloc(sizeof(float) * dim);
    y2 = (float*) malloc(sizeof(float) * dim);
    rep_func_outputs = (float*) malloc(sizeof(float) * 2);
    attr_grads = (float*) malloc(sizeof(float) * n_vertices * dim);
    rep_grads = (float*) malloc(sizeof(float) * n_vertices * dim);
    for(int v=0; v<n_vertices; v++){
        for(d=0; d<dim; d++){
            attr_grads[v * dim + d] = 0;
            rep_grads[v * dim + d] = 0;
        }
    }

    int edge = 0;
    for(int v=0; v<n_vertices; v++){
        for(int nbr=0; nbr<neighbor_counts[v]; nbr++){
            j = head[edge];
            for(d=0; d<dim; d++){
                y1[d] = tail_embedding[v * dim + d];
                y2[d] = head_embedding[j * dim + d];
            }
            dist_squared = sq_euc_dist(y1, y2, dim);

            force = attractive_force_func(
                    normalized,
                    dist_squared,
                    a,
                    b,
                    weights[edge]
            );
            for(d=0; d<dim; d++){
                attr_grads[v * dim + d] -= force * (y1[d] - y2[d]);
            }

            k = rand() % n_vertices;
            for(d=0; d<dim; d++){
                y2[d] = head_embedding[k * dim + d];
            }
            dist_squared = sq_euc_dist(y1, y2, dim);
            repulsive_force_func(
                    rep_func_outputs,
                    normalized,
                    dist_squared,
                    a,
                    b,
                    1.0,
                    0.3 // FIXME -- make avg_weight
            );
            force = rep_func_outputs[0];
            Z += rep_func_outputs[1]; // Z needs to get synchronized if parallel
            for(d=0; d<dim; d++){
                rep_grads[v * dim + d] += force * (y1[d] - y2[d]);
            }

            edge++;
        }
    }
    if(normalized == 1)
        Z = 1;

    for(int v=0; v<n_vertices; v++){
        for(d=0; d<dim; d++){
            index = v * dim + d;
            grad_d = rep_grads[index] / Z - attr_grads[index];

            if(grad_d * all_updates[index] > 0.0)
                gains[index] += 0.2;
            else
                gains[index] *= 0.8;
            gains[index] = clip(gains[index], 0.01, 100);
            grad_d *= gains[index];

            all_updates[index] = grad_d * lr + momentum * 0.9 * all_updates[index];
            head_embedding[index] += all_updates[index];
        }
    }

    free(attr_grads);
    free(rep_grads);
    free(rep_func_outputs);
    free(y1);
    free(y2);
}
