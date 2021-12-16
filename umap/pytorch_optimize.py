import torch

def base_loss(
    attract_dists,
    repel_dists,
    weights,
    average_weight,
    a,
    b
):
    # calculate q(y_i, y_j)
    q_att = 1 / (1 + a * torch.pow(attract_dists, b))
    q_rep = 1 / (1 + a * torch.pow(repel_dists, b))
    q_rep[torch.where(q_rep == 1)] = 0

    # p(x_i, x_j) * log(q(y_i, y_j))
    prob_edge = weights * torch.log(q_att)

    # (1 - p(x_i, x_k)) * log(1 - q(y_i, y_k))
    prob_no_edge = (1 - average_weight) * torch.log(1 - q_rep)

    return -1 * (torch.sum(prob_edge) + torch.sum(prob_no_edge))

def normalized_loss(
    attract_dists,
    repel_dists,
    weights,
    a,
    b
):
    # calculate normalized q(y_i, y_j)
    # weights have already been normalized
    q_att = 1 / (1 + a * torch.pow(attract_dists, b))
    q_att /= torch.sum(q_att)
    q_rep = 1 / (1 + a * torch.pow(repel_dists, b))
    q_rep /= torch.sum(q_rep)

    # p(x_i, x_j) * log(q(y_i, y_j))
    loss = weights * torch.log(q_att)

    return torch.sum(loss)

def squared_dists(x, y):
    return torch.sum(torch.square(x - y), dim=1)

def torch_optimize_layout(
    optimize_method,
    normalization,
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
    alpha,
    verbose=True
):
    n_edges = weights.shape[0]
    dim = head_embedding.shape[1]
    weights = torch.from_numpy(weights)
    head = torch.from_numpy(head).type(torch.long)
    tail = torch.clone(torch.from_numpy(tail).type(torch.long))
    # Perform weight scaling on high-dimensional relationships
    if normalization == 0:
        weights = weights / torch.mean(weights)
        initial_alpha = alpha * 200
    else:
        initial_alpha = alpha

    head_embedding = torch.tensor(head_embedding, requires_grad=True)
    if sym_attraction:
        tail_embedding = torch.tensor(tail_embedding, requires_grad=True)
        optimizer = torch.optim.SGD(
            [{'params': head_embedding}, {'params': tail_embedding}],
            lr=initial_alpha,
            momentum=0.9 * momentum # momentum variable is a 0/1 integer
        )
    else:
        # FIXME - this will NOT let you do .fit() -> .transform()
        tail_embedding = torch.from_numpy(tail_embedding)
        optimizer = torch.optim.SGD(
            [{'params': head_embedding}],
            lr=initial_alpha,
            momentum=0.9 * momentum # momentum variable is a 0/1 integer
        )
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=n_epochs
    )
    average_weight = torch.mean(weights)

    for i_epoch in range(n_epochs):
        # t-SNE early exaggeration
        # if i_epoch == 0 and normalization == 0:
        #     average_weight *= 4
        #     weights *= 4
        # elif i_epoch == 50 and normalization == 0:
        #     weights /= 4
        #     average_weight /= 4

        points = head_embedding[head]
        nearest_neighbors = tail_embedding[tail]
        tail_perm = tail[torch.randperm(tail.nelement())]
        random_points = tail_embedding[tail_perm]

        near_nb_dists = torch.norm(points - nearest_neighbors, dim=1)
        rand_pt_dists = torch.norm(points - random_points, dim=1)

        if normalization == 'umap':
            loss = base_loss(
                near_nb_dists,
                rand_pt_dists,
                weights,
                average_weight,
                a,
                b
            )
        else:
            loss = normalized_loss(
                near_nb_dists,
                rand_pt_dists,
                weights,
                a,
                b
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if verbose and i_epoch % int(n_epochs / 10) == 0:
            print("Completed ", i_epoch, " / ", n_epochs, "epochs")

    return head_embedding.detach().numpy()

