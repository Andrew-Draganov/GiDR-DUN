double fastpow(double a, double b);

float clip(float val, float lower, float upper);
float sq_euc_dist(float* x, float* y, int dim);
float get_lr(float initial_lr, int i_epoch, int n_epochs); 
void print_status(int i_epoch, int n_epochs);

void norm_rep_force(
    float* rep_func_outputs,
    float dist_squared,
    float a,
    float b,
    float cell_size
);
void unnorm_rep_force(
    float* rep_func_outputs,
    float dist_squared,
    float a,
    float b,
    float cell_size,
    float average_weight
);
float umap_attraction_grad(
    float dist_squared, 
    float a,
    float b
);
float umap_repulsion_grad(
    float dist_squared,
    float a,
    float b
);
float kernel_function(
    float dist_squared,
    float a,
    float b
);
float attractive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
);
void repulsive_force_func(
        float* rep_func_outputs,
        int normalized,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight
);
