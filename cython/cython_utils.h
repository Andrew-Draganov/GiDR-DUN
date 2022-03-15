static float clip(float value, float lower, float upper);
static float sq_euc_dist(float* x, float* y, int dim);
static float get_lr(float initial_lr, int i_epoch, int n_epochs) ;
static void print_status(int i_epoch, int n_epochs);
static float umap_repulsion_grad(float dist_squared, float a, float b);
static float kernel_function(float dist_squared, float a, float b);
static float attractive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
);
static void repulsive_force_func(
        float* rep_func_outputs,
        int normalized,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight
);
