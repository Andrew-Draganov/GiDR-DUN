// https://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
#include <stdio.h>
#include <math.h>
#include "fastpow.h"
static double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    if(b == 1.0){
        return a;
    }
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

float my_fmin(float a, float b) {
    return fmin(a, b);
}
float my_fmax(float a, float b) {
    return fmax(a, b);
}

float clip(float val, float lower, float upper) {
    return fmax(lower, fmin(val, upper));
}

float sq_euc_dist(float* x, float* y, int dim) {
    float result = 0.0;
    float diff = 0;
    int i;
    for(i = 0; i < dim; i++)
    {
        diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float get_lr(float initial_lr, int i_epoch, int n_epochs){ 
    return initial_lr * (1.0 - (((float)i_epoch) / ((float)n_epochs)));
}

void print_status(int i_epoch, int n_epochs){
    int print_rate = n_epochs / 10;
    if(i_epoch % print_rate == 0)
        printf("Completed %d / %d epochs\n", i_epoch, n_epochs);
}


float umap_attraction_grad(float dist_squared, float a, float b){
    float grad_scalar = 0.0;
    grad_scalar = 2.0 * a * b * fastpow(dist_squared, b - 1.0);
    grad_scalar /= a * fastpow(dist_squared, b) + 1.0;
    return grad_scalar;
}

float umap_repulsion_grad(float dist_squared, float a, float b){
    float phi_ijZ = 0.0;
    phi_ijZ = 2.0 * b;
    phi_ijZ /= (0.001 + dist_squared) * (a * fastpow(dist_squared, b) + 1);
    return phi_ijZ;
}

float kernel_function(float dist_squared, float a, float b){
    if(b <= 1)
        return 1 / (1 + a * fastpow(dist_squared, b));
    return fastpow(dist_squared, b - 1) / (1 + a * fastpow(dist_squared, b));
}


void unnorm_rep_force(
        float* rep_func_outputs,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight
){
    float kernel, repulsive_force;
    // Realistically, we should use the actual weight on
    //   the edge e_{ik}, but we have not gone through
    //   and calculated this value for each weight. Instead,
    //   we have only calculated them for the nearest neighbors.
    kernel = umap_repulsion_grad(dist_squared, a, b);
    repulsive_force = cell_size * kernel * (1 - average_weight);

    rep_func_outputs[0] = repulsive_force;
    rep_func_outputs[1] = 1; // Z is not gathered in unnormalized setting
}

void norm_rep_force(
        float* rep_func_outputs,
        float dist_squared,
        float a,
        float b,
        float cell_size
){
    float kernel, q_ij, repulsive_force;

    kernel = kernel_function(dist_squared, a, b);
    q_ij = cell_size * kernel; // Collect the q_ij's contributions into Z
    repulsive_force = cell_size * kernel * kernel;

    rep_func_outputs[0] = repulsive_force;
    rep_func_outputs[1] = q_ij;
}

float attractive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
){
    float edge_force;
    if(normalized == 0)
        edge_force = umap_attraction_grad(dist_squared, a, b);
    else
        edge_force = kernel_function(dist_squared, a, b);

    return edge_force * edge_weight;
}

void repulsive_force_func(
        float* rep_func_outputs,
        int normalized,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight
){
    if(normalized == 1)
        norm_rep_force(
            rep_func_outputs,
            dist_squared,
            a,
            b,
            cell_size
        );
    else
        unnorm_rep_force(
            rep_func_outputs,
            dist_squared,
            a,
            b,
            cell_size,
            average_weight
        );
}


float pos_force(
    int normalized,
    float p,
    float q,
    float Z
){
    if(normalized == 1){
        // FIXME - is it faster to get q^2 and then use that for q^3?
        return Z * p * (q * q + 2 * pow(q, 3));
    }
    return p * q * q;
}

float neg_force(
    int normalized,
    float q,
    float Z
){
    if(normalized == 1)
        return Z * (pow(q, 3) * q + 2 * pow(q, 4));
    return pow(q, 3);
}
