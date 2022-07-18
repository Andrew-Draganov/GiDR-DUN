import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def umap_attract_grad(dist_squared, a, b):
    grad_scalar = torch.ones_like(dist_squared)
    grad_scalar *= 2.0 * a * b * torch.pow(dist_squared, b - 1.0)
    grad_scalar /= a * torch.pow(dist_squared, b) + 1.0
    return grad_scalar

def umap_repel_grad(dist_squared, a, b):
    grad_scalar = torch.ones_like(dist_squared) * 2.0 * b
    grad_scalar /= (0.001 + dist_squared) * (a * torch.pow(dist_squared, b) + 1)
    return grad_scalar

def umap_attr_forces(dist_squared, a, b, edge_weights, weight_scalar):
    edge_forces = umap_attract_grad(dist_squared, a, b)
    return edge_forces * edge_weights * weight_scalar

def umap_repel_forces(dist_squared, a, b, average_weight):
    kernels = umap_repel_grad(dist_squared, a, b)
    grads = kernels * (1 - average_weight)
    return (grads, 1)

def tsne_kernel(dist_squared, a, b):
    # torch.pow(x, y) starts with a check for whether y = 1
    #   So it is not a slow-down if b = 1
    if b <= 1:
        return torch.ones_like(dist_squared) / (1 + a * torch.pow(dist_squared, b))
    return torch.pow(dist_squared, b - 1) / (1 + a * torch.pow(dist_squared, b))

def tsne_attr_forces(dist_squared, a, b, edge_weights, weight_scalar):
    edge_forces = tsne_kernel(dist_squared, a, b)
    return edge_forces * edge_weights * weight_scalar

def tsne_repel_forces(dist_squared, a, b, average_weight):
    kernel = tsne_kernel(dist_squared, a, b)
    Z = torch.sum(kernel) # Collect the q_ij's contributions into Z
    grads = kernel * kernel
    return (grads, Z)

def get_lr(initial_lr, i_epoch, n_epochs, normalized):
    if normalized:
        return initial_lr
    return initial_lr * (1.0 - float(i_epoch) / n_epochs)

def squared_dists(x, y):
    return torch.sum(torch.square(x - y), dim=1)

def torch_optimize_frob(
    normalized,
    sym_attraction,
    amplify_grads,
    batch_size,
    embedding,
    head,
    tail,
    weights,
    forces,
    gains,
    n_epochs,
    n_vertices,
    a,
    b,
    initial_lr,
    verbose=True,
    **kwargs
):
    if normalized:
        raise ValueError('Torch frobenius norm optimization does not work in normalized setting :(')

    n_edges = weights.shape[0]
    dim = int(embedding.shape[1])
    Z = 0
    step_forces = torch.zeros_like(embedding).cuda()
    attr_forces = torch.zeros_like(embedding).cuda()
    rep_forces = torch.zeros_like(embedding).cuda()

    # Gradient descent loop
    for i_epoch in tqdm(range(n_epochs)):
        # t-SNE early exaggeration
        if i_epoch < 250:
            weight_scalar = 4
        else:
            weight_scalar = 1
        
        attr_forces *= 0
        rep_forces *= 0

        # Get points in low-dim whose high-dim analogs are nearest neighbors
        points = embedding[head]
        nearest_neighbors = embedding[tail]
        attr_vectors = points - nearest_neighbors
        near_nb_dists = squared_dists(points, nearest_neighbors)
        Q1 = tsne_kernel(near_nb_dists, a, b)

        tail_perms = torch.randperm(n_edges)
        tail_perms = tail_perms % n_vertices
        random_points = embedding[tail_perms]
        rep_vectors = points - random_points
        rand_pt_dists = squared_dists(points, random_points)
        Q2 = tsne_kernel(rand_pt_dists, a, b)

        # Calculate attractive forces
        attr_grads = weights * Q1 * Q1
        attr_grads *= weight_scalar
        attr_grads = torch.unsqueeze(attr_grads, 1) * attr_vectors
        attr_forces = attr_forces.index_add(0, head, attr_grads)
        if sym_attraction:
            attr_forces = attr_forces.index_add(0, tail, -1 * attr_grads)
        attr_forces *= 4 * a * b

        rep_grads = Q2 * Q2 * Q2
        rep_grads = torch.unsqueeze(rep_grads, 1) * rep_vectors
        rep_forces = rep_forces.index_add(0, head, rep_grads)
        rep_forces *= 4 * a * b

        # Gradient descent
        lr = get_lr(initial_lr, i_epoch, n_epochs, normalized)
        step_forces = rep_forces - attr_forces
        inc_indices = forces * step_forces > 0.0
        gains[inc_indices] += 0.2
        gains[~inc_indices] *= 0.8
        step_forces *= gains
        forces = forces * 0.9 * float(amplify_grads) + step_forces * lr
        forces = torch.clamp(forces, -1, 1)
        embedding += forces

    return embedding.cpu().numpy()

def torch_optimize_kl(
    normalized,
    sym_attraction,
    amplify_grads,
    batch_size,
    embedding,
    head,
    tail,
    weights,
    forces,
    gains,
    n_epochs,
    n_vertices,
    a,
    b,
    initial_lr,
    verbose=True,
    **kwargs
):
    n_edges = weights.shape[0]

    # Perform weight scaling on high-dimensional relationships
    if normalized:
        weights = weights / torch.sum(weights)
    average_weight = torch.mean(weights)

    if normalized:
        attr_force_func = tsne_attr_forces
        rep_force_func = tsne_repel_forces
    else:
        attr_force_func = umap_attr_forces
        rep_force_func = umap_repel_forces
    attr_forces = torch.zeros_like(embedding).cuda()
    rep_forces = torch.zeros_like(embedding).cuda()

    # Gradient descent loop
    for i_epoch in tqdm(range(n_epochs)):
        # t-SNE early exaggeration
        if i_epoch < 250:
            weight_scalar = 4
        else:
            weight_scalar = 1

        attr_forces *= 0
        rep_forces *= 0
        Z = 0

        # Get points in low-dim whose high-dim analogs are nearest neighbors
        points = embedding[head]
        nearest_neighbors = embedding[tail]
        attr_vectors = points - nearest_neighbors
        near_nb_dists = squared_dists(points, nearest_neighbors)

        tail_perms = torch.randperm(n_edges)
        tail_perms = tail_perms % n_vertices
        random_points = embedding[tail_perms]
        rep_vectors = points - random_points
        rand_pt_dists = squared_dists(points, random_points)

        # Calculate attractive forces
        attr_grads = attr_force_func(near_nb_dists, a, b, weights, weight_scalar)
        attr_grads = torch.clamp(torch.unsqueeze(attr_grads, 1) * attr_vectors, -4, 4)
        attr_forces = attr_forces.index_add(0, head, attr_grads)
        if sym_attraction:
            attr_forces = attr_forces.index_add(0, tail, -1 * attr_grads)

        # Calculate repulsive forces
        rep_grads, Z = rep_force_func(rand_pt_dists, a, b, average_weight)
        rep_grads = torch.clamp(torch.unsqueeze(rep_grads, 1) * rep_vectors, -4, 4)
        rep_forces = rep_forces.index_add(0, head, rep_grads)

        if normalized:
            # p_{ij} in the attractive forces is normalized by the sum of the weights,
            #     so we need to do the same to q_{ij} in the repulsive forces
            rep_forces *= 4 * a * b / Z
            attr_forces *= 4 * a * b

        # Momentum gradient descent
        lr = get_lr(initial_lr, i_epoch, n_epochs, normalized)
        step_forces = rep_forces - attr_forces
        inc_indices = forces * step_forces > 0.0
        gains[inc_indices] += 0.2
        gains[~inc_indices] *= 0.8
        step_forces *= gains
        forces = forces * 0.9 * float(amplify_grads) + step_forces * lr
        forces = torch.clamp(forces, -1, 1)
        embedding += forces

    return embedding.cpu().numpy()

def torch_optimize_layout(
    optimize_method,
    normalized,
    frob,
    sym_attraction,
    amplify_grads,
    head_embedding,
    head,
    tail,
    weights,
    n_epochs,
    n_vertices,
    a,
    b,
    initial_lr,
    verbose=True,
    batch_size=32,
    **kwargs
):
    with torch.no_grad():
        a = torch.tensor(a)
        b = torch.tensor(b)

        # FIXME FIXME
        # CURRENT ATTEMPT -- go to cuda 11.0 rather than 11.3???
        # Just ran the install before leaving
        weights = torch.from_numpy(weights).type(torch.float).cuda()
        head = torch.from_numpy(head).type(torch.long).cuda()
        tail = torch.clone(torch.from_numpy(tail).type(torch.long)).cuda()
        head_embedding = torch.from_numpy(head_embedding).type(torch.float).cuda()
        forces = torch.zeros_like(head_embedding).type(torch.float).cuda()
        gains = torch.ones_like(head_embedding).type(torch.float).cuda()

        if frob:
            optimization_fn = torch_optimize_frob
        else:
            optimization_fn = torch_optimize_kl

        head_embedding = optimization_fn(
            normalized,
            sym_attraction,
            amplify_grads,
            batch_size,
            head_embedding,
            head,
            tail,
            weights,
            forces,
            gains,
            n_epochs,
            n_vertices,
            a,
            b,
            initial_lr,
            verbose=verbose,
        )

    return head_embedding
