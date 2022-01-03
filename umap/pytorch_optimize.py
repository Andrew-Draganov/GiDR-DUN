import torch

@torch.jit.script
def umap_attract_grad(dist_squared, a, b):
    grad_scalar = torch.ones_like(dist_squared)
    grad_scalar *= -2.0 * a * b * torch.pow(dist_squared, b - 1.0)
    grad_scalar /= a * torch.pow(dist_squared, b) + 1.0
    return grad_scalar

@torch.jit.script
def umap_repel_grad(dist_squared, a, b):
    grad_scalar = torch.ones_like(dist_squared) * 2.0 * b
    grad_scalar /= (0.001 + dist_squared) * (a * torch.pow(dist_squared, b) + 1)
    return grad_scalar

@torch.jit.script
def umap_attr_forces(dist_squared, a, b, edge_weights):
    edge_forces = umap_attract_grad(dist_squared, a, b)
    return edge_forces * edge_weights

@torch.jit.script
def umap_repel_forces(dist_squared, a, b, average_weight):
    kernels = umap_repel_grad(dist_squared, a, b)
    grads = kernels * (1 - average_weight)
    return (grads, 0)


@torch.jit.script
def tsne_kernel(dist_squared, a, b):
    if b <= 1:
        return torch.ones_like(dist_squared) / (1 + a * torch.pow(dist_squared, b))
    return torch.pow(dist_squared, b - 1) / (1 + a * torch.pow(dist_squared, b))

@torch.jit.script
def tsne_attr_forces(dist_squared, a, b, edge_weights):
    edge_forces = tsne_kernel(dist_squared, a, b)
    return edge_forces * edge_weights

@torch.jit.script
def tsne_repel_forces(dist_squared, a, b, average_weight):
    kernel = tsne_kernel(dist_squared, a, b)
    Z = torch.sum(kernel) # Collect the q_ij's contributions into Z
    grads = kernel * kernel
    return (grads, Z)

@torch.jit.script
def squared_dists(x, y):
    return torch.sum(torch.square(x - y), dim=1)

def torch_optimize_layout(
    optimize_method,
    normalized,
    sym_attraction,
    momentum,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    initial_lr,
    negative_sample_rate,
    verbose=True,
    **kwargs
):
    a = torch.tensor(a)
    b = torch.tensor(b)
    n_edges = weights.shape[0]
    n_points = head_embedding.shape[0]
    weights = torch.from_numpy(weights)
    head = torch.from_numpy(head).type(torch.long)
    # FIXME - is clone necessary to avoid pass-by-reference concerns?
    tail = torch.clone(torch.from_numpy(tail).type(torch.long))
    head_embedding = torch.from_numpy(head_embedding)
    tail_embedding = torch.from_numpy(tail_embedding)
    forces = torch.zeros_like(head_embedding)

    # Perform weight scaling on high-dimensional relationships
    if normalized:
        weights = weights / torch.sum(weights)
        initial_lr *= 200
    average_weight = torch.mean(weights)

    if normalized:
        attr_force_func = tsne_attr_forces
        rep_force_func = tsne_repel_forces
    else:
        attr_force_func = umap_attr_forces
        rep_force_func = umap_repel_forces
    attr_forces = torch.zeros_like(head_embedding)
    rep_forces = torch.zeros_like(head_embedding)

    # Gradient descent loop
    for i_epoch in range(n_epochs):
        # t-SNE early exaggeration
        if i_epoch == 0 and normalized:
            average_weight *= 4
            weights *= 4
        elif i_epoch == 50 and normalized:
            weights /= 4
            average_weight /= 4

        # Get points in low-dim whose high-dim analogs are nearest neighbors
        points = head_embedding[head]
        nearest_neighbors = tail_embedding[tail]
        attr_vectors = points - nearest_neighbors

        # Get n_edges worth of random points to perform repulsions
        tail_perm = torch.randperm(n_edges)
        tail_perm = tail_perm % n_points
        random_points = tail_embedding[tail_perm]
        rep_vectors = points - random_points

        # Squared Euclidean distance between attr and rep points
        near_nb_dists = squared_dists(points, nearest_neighbors)
        rand_pt_dists = squared_dists(points, random_points)

        # Calculate attractive forces
        attr_forces *= 0
        attr_grads = attr_force_func(near_nb_dists, a, b, weights)
        attr_grads = torch.clamp(torch.unsqueeze(attr_grads, 1) * attr_vectors, -4, 4)
        attr_forces = attr_forces.index_add(0, head, attr_grads)
        if sym_attraction:
            attr_forces = attr_forces.index_add(0, tail, -1 * attr_grads)

        # Calculate repulsive forces
        rep_forces *= 0
        rep_grads, Z = rep_force_func(rand_pt_dists, a, b, average_weight)
        rep_grads = torch.clamp(torch.unsqueeze(rep_grads, 1) * rep_vectors, -4, 4)
        rep_forces = rep_forces.index_add(0, head, rep_grads)

        if normalized:
            # p_{ij} in the attractive forces is normalized by the sum of the weights,
            #     so we need to do the same to q_{ij} in the repulsive forces
            rep_forces *= 4 * a * b / Z
            attr_forces *= -4 * a * b

        # Momentum gradient descent
        lr = initial_lr * (1.0 - float(i_epoch) / n_epochs)
        forces = forces * 0.9 * float(momentum) + (attr_forces + rep_forces)
        head_embedding += forces * lr

        if verbose and i_epoch % int(n_epochs / 10) == 0:
            print("Completed ", i_epoch, " / ", n_epochs, "epochs")

    return head_embedding.detach().numpy()
