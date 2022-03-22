#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// https://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
static double fast_pow(double a, double b) {
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

static float clip(float val, float lower, float upper) {
    return fmax(lower, fmin(val, upper));
}

static float sq_euc_dist(float* x, float* y, int dim) {
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

static float ang_dist(float* x, float* y, int dim) {
    // cosine distance between vectors x and y
    float result = 0.0;
    float x_len  = 0.0;
    float y_len  = 0.0;
    float eps = 0.0001;
    for(int i=0; i<dim; i++){
        result += x[i] * y[i];
        x_len += x[i] * x[i];
        y_len += y[i] * y[i];
    }
    if(x_len < eps || y_len < eps)
        return 1;
    return result / (sqrt(x_len * y_len));
}

static float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads){ 
    if(amplify_grads == 1)
        return initial_lr;
    return initial_lr * (1.0 - (((float)i_epoch) / ((float)n_epochs)));
}

static void print_status(int i_epoch, int n_epochs){
    int print_rate = n_epochs / 10;
    if(i_epoch % print_rate == 0)
        printf("Completed %d / %d epochs\n", i_epoch, n_epochs);
}

float get_avg_weight(float* weights, int n_edges){
    float average_weight = 0.0;
    for (int i=0; i < n_edges; i++)
        average_weight += weights[i];
    average_weight /= n_edges;
    return average_weight;
}

static float umap_attr_scalar(float dist, float a, float b){
    float grad_scalar = 0.0;
    grad_scalar = 2.0 * a * b * fast_pow(dist, b - 1.0);
    grad_scalar /= a * fast_pow(dist, b) + 1.0;
    return grad_scalar;
}

static float umap_rep_scalar(float dist, float a, float b){
    float phi_ijZ = 0.0;
    phi_ijZ = 2.0 * b;
    phi_ijZ /= (0.001 + dist) * (a * fast_pow(dist, b) + 1);
    return phi_ijZ;
}

static float kernel_function(float dist, float a, float b){
    if(b <= 1)
        return 1 / (1 + a * fast_pow(dist, b));
    return fast_pow(dist, b - 1) / (1 + a * fast_pow(dist, b));
}


static float kl_attr_force(float p, float q){
    return p * q;
}

static float kl_rep_force(int normalized, float q, float avg_weight){
    if(normalized)
        return q * q;
    // Realistically, we should use the actual weight on
    //   the edge e_{ik}, but we have not gone through
    //   and calculated this value for each weight. Instead,
    //   we have only calculated them for the nearest neighbors.
    return q * (1 - avg_weight);
}

static float frob_attr_force(int normalized, float p, float q){
    if(normalized){
        // FIXME - is it faster to get q^2 and then use that for q^3?
        // FIXME - took out a Z scalar from this
        return  p * (q * q + 2 * fast_pow(q, 3));
    }
    return p * q * q;
}

static float frob_rep_force(int normalized, float q){
    if(normalized)
        return fast_pow(q, 3) + 2 * fast_pow(q, 4);
    return fast_pow(q, 3);
}

static float attractive_force_func(
        int normalized,
        int frob,
        float dist,
        float a,
        float b,
        float edge_weight
){
    float q;
    if(normalized || frob)
        q = kernel_function(dist, a, b);
    else
        q = umap_attr_scalar(dist, a, b);

    if(frob)
        return frob_attr_force(normalized, edge_weight, q);
    else
        return kl_attr_force(edge_weight, q);
}

static void repulsive_force_func(
        float* rep_func_outputs,
        int normalized,
        int frob,
        float dist,
        float a,
        float b,
        float cell_size,
        float avg_weight
){
    float q;
    if(normalized || frob)
        q = kernel_function(dist, a, b);
    else
        q = umap_rep_scalar(dist, a, b);

    if(frob)
        rep_func_outputs[0] = frob_rep_force(normalized, q);
    else
        rep_func_outputs[0] = kl_rep_force(normalized, q, avg_weight);
    rep_func_outputs[0] *= cell_size;

    if(normalized)
        rep_func_outputs[1] = q * cell_size;
    else
        // Do not collect Z in unnormalized case
        rep_func_outputs[1] = 0;
}
