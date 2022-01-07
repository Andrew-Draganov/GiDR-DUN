import torch
from torch.utils.data import Dataset, DataLoader

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
    # torch.pow(x, y) returns x if y = 1
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

class NNGraphDataset(Dataset):
    def __init__(
        self,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights
    ):
        """
        This dataset will operate on the edges in the nearest neighbor graph
        that is made for the high-dimensional dataset
        """
        self.head_embedding = head_embedding
        self.tail_embedding = tail_embedding
        self.head = head
        self.tail = tail
        self.weights = weights

    def __len__(self):
        """ len is the number of edges in the nearest neighbor graph """
        return len(self.weights)

    def __getitem__(self, index):
        # Select sample
        y_i = self.head_embedding[self.head[index]]

        # Get y_i's y_j such that (x_i, x_j) are nearest neighbors
        nearest_neighbor = self.tail_embedding[self.tail[index]]

        # Get random point to repel from
        random_point = torch.squeeze(self.tail_embedding[
            torch.randint(
                low=0,
                high=len(self.tail_embedding),
                size=(1,)
            )
        ])

        # Weight between y_i and its relative nearest neighbor
        weight = self.weights[index]

        ret_tuple = (
            y_i,
            nearest_neighbor,
            random_point,
            weight,
            self.head[index],
            self.tail[index]
        )

        return ret_tuple

def torch_optimize_batched(
    normalized,
    sym_attraction,
    momentum,
    batch_size,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    forces,
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

    dataset = NNGraphDataset(head_embedding, tail_embedding, head, tail, weights)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    generator = iter(data_loader)

    # Gradient descent loop
    for i_batch in range(n_epochs):
        try:
            batch = next(generator)
        except StopIteration:
            generator = iter(data_loader)
            batch = next(generator)

        # Get batch of attraction and repulsion directions
        points = batch[0]
        nearest_neighbors = batch[1]
        random_points = batch[2]
        weights = batch[3]
        head_batch = batch[4]
        tail_batch = batch[5]
        attr_vectors = points - nearest_neighbors
        rep_vectors = points - random_points

        # t-SNE early exaggeration
        if i_batch == 0 and normalized:
            average_weight *= 4
            weights *= 4
        elif i_batch == 50 and normalized:
            weights /= 4
            average_weight /= 4

        # Squared Euclidean distance between attr and rep points
        near_nb_dists = squared_dists(points, nearest_neighbors)
        rand_pt_dists = squared_dists(points, random_points)

        # Calculate attractive forces
        attr_forces *= 0
        attr_grads = attr_force_func(near_nb_dists, a, b, weights)
        attr_grads = torch.clamp(torch.unsqueeze(attr_grads, 1) * attr_vectors, -4, 4)
        attr_forces = attr_forces.index_add(0, head_batch, attr_grads)
        if sym_attraction:
            attr_forces = attr_forces.index_add(0, tail_batch, -1 * attr_grads)

        # Calculate repulsive forces
        rep_forces *= 0
        rep_grads, Z = rep_force_func(rand_pt_dists, a, b, average_weight)
        rep_grads = torch.clamp(torch.unsqueeze(rep_grads, 1) * rep_vectors, -4, 4)
        rep_forces = rep_forces.index_add(0, head_batch, rep_grads)

        if normalized:
            # p_{ij} in the attractive forces is normalized by the sum of the weights,
            #     so we need to do the same to q_{ij} in the repulsive forces
            rep_forces *= 4 * a * b / Z
            attr_forces *= -4 * a * b

        # Momentum gradient descent
        lr = initial_lr * (1.0 - float(i_batch) / n_epochs)
        forces[head_batch] = forces[head_batch] * 0.9 * float(momentum)

        # FIXME - indexing probably not necessary here
        forces += attr_forces + rep_forces
        head_embedding[head_batch] += forces[head_batch] * lr

        if verbose and i_batch % int(n_epochs / 10) == 0:
            print("Completed ", i_batch, " / ", n_epochs, "epochs")

    return head_embedding.detach().numpy()

def get_num_repulsions(attr_count, ratio_count, ratio):
    if ratio == 1:
        return 1, 1, 1

    if ratio > 1:
        if int(ratio_count) > attr_count:
            num_repulsions = 1 + int(ratio_count) - attr_count
            ratio_count -= (num_repulsions - 1)
        else:
            num_repulsions = 1
    elif ratio < 1:
        if ratio_count >= 1:
            num_repulsions = 1
            ratio_count -= 1
        else:
            num_repulsions = 0
    elif ratio < 0:
        raise ValueError("Cannot have a negative attr-repel ratio.")

    return num_repulsions, attr_count, ratio_count


def torch_optimize_full(
    normalized,
    sym_attraction,
    momentum,
    batch_size,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    forces,
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
    attr_count = 0
    ratio_count = 0
    ratio = 1.5
    for i_epoch in range(n_epochs):
        # t-SNE early exaggeration
        if i_epoch == 0 and normalized:
            average_weight *= 4
            weights *= 4
        elif i_epoch == 50 and normalized:
            weights /= 4
            average_weight /= 4

        attr_forces *= 0
        rep_forces *= 0

        # Get points in low-dim whose high-dim analogs are nearest neighbors
        points = head_embedding[head]
        nearest_neighbors = tail_embedding[tail]
        attr_vectors = points - nearest_neighbors
        near_nb_dists = squared_dists(points, nearest_neighbors)

        # Calculate attractive forces
        attr_grads = attr_force_func(near_nb_dists, a, b, weights)
        attr_grads = torch.clamp(torch.unsqueeze(attr_grads, 1) * attr_vectors, -4, 4)
        attr_forces = attr_forces.index_add(0, head, attr_grads)
        if sym_attraction:
            attr_forces = attr_forces.index_add(0, tail, -1 * attr_grads)

        num_repulsions, attr_count, ratio_count = get_num_repulsions(
            attr_count,
            ratio_count,
            ratio
        )

        if num_repulsions > 0:
            # Get num_repulsions worth of random points to perform repulsions
            tail_perms = [torch.randperm(n_edges) for i in range(num_repulsions)]
            tail_perms = [tail_perm % n_vertices for tail_perm in tail_perms]
            random_points = [tail_embedding[tail_perm] for tail_perm in tail_perms]
            random_points = torch.stack(random_points, dim=0)
            random_points = torch.mean(random_points, dim=0)
            rep_vectors = points - random_points

            # Squared Euclidean distance between attr and rep points
            rand_pt_dists = squared_dists(points, random_points)

            # Calculate repulsive forces
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

        attr_count += 1
        ratio_count += ratio

    return head_embedding.detach().numpy()

def torch_optimize_layout(
    optimize_method,
    normalized,
    sym_attraction,
    momentum,
    batch_size,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    n_epochs,
    n_vertices,
    a,
    b,
    initial_lr,
    verbose=True,
    **kwargs
):
    a = torch.tensor(a)
    b = torch.tensor(b)
    weights = torch.from_numpy(weights)
    head = torch.from_numpy(head).type(torch.long)
    # FIXME - is clone necessary to avoid pass-by-reference concerns?
    tail = torch.clone(torch.from_numpy(tail).type(torch.long))
    head_embedding = torch.from_numpy(head_embedding)
    tail_embedding = torch.from_numpy(tail_embedding)
    forces = torch.zeros_like(head_embedding)
    if 'sgd' in optimize_method:
        optimization_fn = torch_optimize_batched
    else:
        optimization_fn = torch_optimize_full

    embedding = optimization_fn(
        normalized,
        sym_attraction,
        momentum,
        batch_size,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        forces,
        n_epochs,
        n_vertices,
        a,
        b,
        initial_lr,
        verbose=verbose,
    )

    return embedding
